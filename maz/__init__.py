import sys
import random
import time
import base64
import os
import glob
import threading
from queue import Queue, Empty
import joblib
from concurrent.futures import ThreadPoolExecutor, as_completed
import orjson
import tqdm


def load_pkl(logger, cache_path: str):
    logger.info(f"Loading static cache from [{cache_path}]")
    s = time.time()
    with open(cache_path, 'rb') as fin:
        data = joblib.load(fin)
    logger.info(
        f"Loaded static cache from [{cache_path}] with {len(data)} samples in [{time.time() - s:.2f}] seconds")
    return data


class MazDataset:
    json_glob = "__woec*.json"

    def __init__(self, config, mode, logger, seed=None):
        from data.imaug import transform, create_operators

        self.transform = transform
        self.logger = logger
        self.mode = mode.lower()
        self.need_reset = False

        global_config = config["Global"]
        dataset_config = config[mode]["dataset"]
        self.static_mode_load = dataset_config.get("static_mode_load", False)
        self.static_mode_save = dataset_config.get("static_mode_save", False)
        self.static_mode_save_final = dataset_config.get("static_mode_save_final", None)
        self.shuffle = True
        self.ratio = dataset_config.get("ratio", None)
        self.dynamic_cache = dataset_config.get("dynamic_cache", False)
        if self.dynamic_cache and (self.static_mode_save or self.static_mode_load):
            self.logger.critical("Static mode and dynamic cache cannot be used together")
            sys.exit(1)

        self.data_dir = dataset_config["data_dir"]
        if isinstance(self.data_dir, str):
            self.data_dir = [self.data_dir]
        self.seed = seed

        self.ops = create_operators(dataset_config["transforms"], global_config)
        self.data = []
        self._dcache = {}
        self._dcache_hit = [0, 0]

        # special case
        if self.static_mode_save_final and os.path.exists(self.static_mode_save_final):
            self.data = load_pkl(self.logger, self.static_mode_save_final)
            self.data_dir = []
            self.shuffle = False
            self.ratio = None

        for dt_folder in self.data_dir:
            if os.path.isfile(dt_folder) and dt_folder.endswith(".json"):
                woec_jsons = [dt_folder]
            else:
                woec_jsons = glob.glob(os.path.join(dt_folder, MazDataset.json_glob))
            if len(woec_jsons) == 0:
                logger.critical(f"Missing woec json files in [{dt_folder}]")
                sys.exit(1)

            limit_d = dataset_config.get("max_limit_per_dataset",  {})
            limit_dataset_all = limit_d.get("*", 0)

            for woec_json in woec_jsons:
                if any(excl in woec_json for excl in dataset_config.get("data_dir_exclude", [])):
                    logger.info(f"Skipping [{woec_json}] due to exclusion [data_dir_exclude]")
                    continue

                data = []

                cache_suffix = f".paddle.{self.mode.upper()}.pkl"
                cache_path = f"{woec_json}{cache_suffix}"
                if self.static_mode_load:
                    if not os.path.exists(cache_path):
                        if "static_alt_data_dir" in dataset_config:
                            alt_cache_path = os.path.join(
                                dataset_config["static_alt_data_dir"],
                                str(os.path.basename(woec_json) + {cache_suffix}),
                            )
                            if os.path.exists(alt_cache_path):
                                cache_path = alt_cache_path
                    if os.path.exists(cache_path):
                        data = load_pkl(self.logger, cache_path)
                    else:
                        self.logger.warning(f"Static cache not found in [{cache_path}]")

                if len(data) == 0:
                    # load dataset
                    data = self.load_from_json(woec_json)

                if self.static_mode_save and not os.path.exists(cache_path):
                    max_workers = dataset_config.get("static_max_workers", 8)
                    data = self.prepare_for_train(data, max_workers=max_workers)
                    joblib.dump(data, cache_path)
                    self.logger.info(f"Saved static cache to [{cache_path}]")

                # if we load multiple datasets and we want to limit the number of samples
                # e.g., because of Eval
                limit_dataset = max(limit_dataset_all,
                                    limit_d.get(os.path.basename(woec_json), 0))
                if limit_dataset > 0:
                    random.seed(self.seed)
                    random.shuffle(data)
                    data = data[:limit_dataset]
                    logger.info(f"Limiting [{woec_json}] to {len(data)} samples")

                # finally, add it
                self.data += data

        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(self.data)
        if self.ratio:
            s, e = self.ratio
            s = int(s * len(self.data))
            e = int(e * len(self.data))
            self.data = self.data[s:e]

        if self.static_mode_save_final is not None and not os.path.exists(self.static_mode_save_final):
            self.logger.warning(f"Static mode save final path [{self.static_mode_save_final}] does not exist, saving to it")
            max_workers = dataset_config.get("static_max_workers", 8)
            data = self.prepare_for_train(self.data, max_workers=max_workers)
            joblib.dump(data, self.static_mode_save_final)
            self.logger.info(f"Saved static cache to [{self.static_mode_save_final}]")
            sys.exit(0)

        self._last_idx = -1
        self.logger.info(f"Ratio: {dataset_config.get('ratio', None)}, final size: [{len(self)}]")
        if self.dynamic_cache:
            self.logger.info(f"Starting dynamic cache with [{len(self)}] samples")
            self._dynamic_mode_verbose = dataset_config.get("dynamic_mode_verbose", 0)
            self._cache_lock = threading.Lock()
            self._stop_event = threading.Event()
            self._preload_queue = Queue()
            self._preload_size = dataset_config.get("dynamic_cache_prefetch", 2048)
            self._preload_cursor = 2 * self._preload_size
            self._preload_thread_count = dataset_config.get("dynamic_cache_workers", 4)
            self._preload_threads = []
            # first batch
            for i in range(0, self._preload_cursor):
                self._preload_queue.put(i)
            for _ in range(self._preload_thread_count):
                t = threading.Thread(target=self._preload_worker, daemon=True)
                self._preload_threads.append(t)
                t.start()
            self._queue_feeder_thread = threading.Thread(target=self._preload_feeder_loop, daemon=True)
            self._queue_feeder_thread.start()

    def _safe_process_row(self, idx_row):
        idx, row = idx_row
        try:
            # already done - can happen with static and static final combined
            if not isinstance(row[0], str):
                return idx, row
            outs = self.row2outs(row)
            if outs is None:
                self.logger.debug(f"Transform failed for [{row[0]}]")
            return idx, outs
        except Exception as e:
            self.logger.warning(f"Exception processing row {idx_row}: {e}")
            return idx, None

    def row2outs(self, row):
        gt, buf, paddle_score, img_id = row
        data = {"image": buf, "label": gt, "img_path": img_id, "ext_data": []}
        return self.transform(data, self.ops)

    def load_from_json(self, file_str):
        self.logger.info(f"Loading [{file_str}]")

        with open(file_str, 'r', encoding="utf-8") as fin:
            data = orjson.loads(fin.read())["data"]
        ret = []
        for d in tqdm.tqdm(data):
            gt_score = d["gt"]
            confs = d["confs"]
            gt = d["gttext"]
            img_id = d["img"]
            if "img_data" not in d:
                raise NotImplementedError("img_data not in json")
            if d["img_data"] is None:
                self.logger.critical(f"Missing img_data in [{d}]")
                continue
            buf = base64.b64decode(d["img_data"])
            row = (gt, buf, gt_score[2], img_id)
            ret.append(row)
        return ret

    def __getitem__(self, idx):
        try:
            if self.dynamic_cache:
                outs = self._dcache.get(idx, None)
                if outs is not None:
                    self._dcache_hit[0] += 1
                    return outs
                self._dcache_hit[1] += 1

            row = self.data[idx]
            if not isinstance(row[0], str):
                return row

            outs = self.row2outs(row)
            # e.g., max text length exceeded - select random one
            while outs is None:
                self.logger.debug(f"Transform failed for [{row[0]}]")
                outs = self.__getitem__(random.randint(0, len(self) - 1))
            return outs
        finally:
            self._last_idx = idx

    def __len__(self):
        return len(self.data)

    def prepare_for_train(self, data, max_workers: int):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._safe_process_row, (idx, row))
                       for idx, row in enumerate(data)]
            new_data = [None] * len(data)

            for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
                try:
                    idx, result = future.result()
                    if result is not None:
                        new_data[idx] = result
                except Exception as e:
                    self.logger.exception(f"Future failed: {e}")
            return [x for x in new_data if x is not None]

    def _preload_worker(self):
        while not self._stop_event.is_set():
            try:
                idx = self._preload_queue.get(timeout=0.1)
            except Empty:
                time.sleep(0.1)
                continue

            if idx >= len(self.data):
                continue
            try:
                row = self.data[idx]
                if not isinstance(row[0], str):
                    continue
                result = self.row2outs(row)
                if result is None:
                    continue
                with self._cache_lock:
                    self._dcache[idx] = result
            except Exception as e:
                self.logger.debug(f"Preload exception at idx={idx}: {e}")

    def _preload_feeder_loop(self):
        keep_behind = 16
        sleep_time = 0.25
        epoch = 0

        if self._dynamic_mode_verbose == 0:
            log_N = 5000
            clear_N = 1000
        else:
            log_N = 500
            clear_N = 1000

        id_str = f"dyn cache:{self.mode}: "

        while not self._stop_event.is_set():
            epoch += 1
            last_idx = self._last_idx
            cursor = self._preload_cursor

            if epoch % log_N == 0:
                self.logger.info(
                    f"{id_str}Preload: {self._preload_cursor:5d} "
                    f"last fetched idx: {last_idx:5d} "
                    f"cache size: {len(self._dcache):5d} "
                    f"queue: {self._preload_queue.qsize():5d} "
                    f"epoch: {epoch:5d} "
                    f"cache hit: {self._dcache_hit[0]:5d} "
                    f"cache miss: {self._dcache_hit[1]:5d} "
                )

            # If we're near the end, reset
            if last_idx >= len(self.data) - 512:
                self.logger.info(f"{id_str}Near end of dataset, resetting preload state")
                with self._cache_lock:
                    self._preload_queue.queue.clear()
                fill_to = self._preload_size * 2
                for i in range(0, fill_to):
                    self._preload_queue.put(i)
                self._preload_cursor = fill_to
                while self._last_idx > len(self.data) / 2:
                    time.sleep(sleep_time)
                continue

            # we can clear all fetched results as they are obsolete now
            if last_idx > cursor:
                # clear _preload_queue queue
                with self._cache_lock:
                    self._preload_queue.queue.clear()
                    self._dcache.clear()
                new_cursor = max(last_idx, self._last_idx) + 2 * self._preload_size
                self.logger.info(f"{id_str}Preload cursor reset to {cursor} -> {new_cursor}"
                                 f" [last idx: {self._last_idx}]")
                cursor = new_cursor

            enough_in_queue = self._preload_queue.qsize() > 5 * self._preload_size
            enough_prefetched = len(self._dcache) > 5 * self._preload_size
            if not enough_in_queue and not enough_prefetched:
                for i in range(cursor, min(cursor + self._preload_size, len(self.data))):
                    self._preload_queue.put(i)
                self._preload_cursor += self._preload_size

            # Feed new indices into the queue
            if epoch % clear_N == 0:
                min_to_keep = max(0, last_idx - keep_behind)
                to_delete = [k for k in self._dcache if k < min_to_keep]
                with self._cache_lock:
                    for k in to_delete:
                        del self._dcache[k]
                if len(to_delete) > 0:
                    self.logger.info(f"{id_str}Deleted {len(to_delete)} old cache entries")

            time.sleep(sleep_time)

    def __del__(self):
        if self.dynamic_cache:
            self.logger.info("Shutting down preloader threads...")
            self._stop_event.set()
            for t in self._preload_threads:
                t.join()
            self._queue_feeder_thread.join()
            self.logger.info("Preloader shutdown complete.")