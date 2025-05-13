import sys
import os
import glob
import random
import cv2
import orjson


class MazDatasetRaw:
        json_glob = "__dataset.words.*.json"

        def __init__(self, config, mode, logger, seed=None):
            from data.imaug import transform, create_operators

            self.transform = transform
            self.logger = logger
            self.mode = mode.lower()
            self.need_reset = False

            global_config = config["Global"]
            dataset_config = config[mode]["dataset"]
            self.ratio = dataset_config.get("ratio", None)
            self.data_dir = dataset_config["data_dir"]
            if isinstance(self.data_dir, str):
                self.data_dir = [self.data_dir]
            self.seed = seed
            self.shuffle = True if self.ratio is None else False

            self.ops = create_operators(dataset_config["transforms"], global_config)
            self.data = []

            for dt_folder in self.data_dir:
                if os.path.isfile(dt_folder) and dt_folder.endswith(".json"):
                    jsons = [dt_folder]
                else:
                    jsons = glob.glob(os.path.join(dt_folder, MazDatasetRaw.json_glob))
                if len(jsons) == 0:
                    logger.critical(f"Missing dataset json files in [{dt_folder}]")
                    sys.exit(1)

                limit_d = dataset_config.get("max_limit_per_dataset", {})
                limit_dataset_all = limit_d.get("*", 0)

                for js_file in jsons:
                    if any(excl in js_file for excl in dataset_config.get("data_dir_exclude", [])):
                        logger.info(f"Skipping [{js_file}] due to exclusion [data_dir_exclude]")
                        continue

                    data = self.load_from_json(js_file, os.path.dirname(js_file))

                    # if we load multiple datasets and we want to limit the number of samples
                    # e.g., because of Eval
                    limit_dataset = max(limit_dataset_all,
                                        limit_d.get(os.path.basename(js_file), 0))
                    if limit_dataset > 0:
                        random.seed(self.seed)
                        random.shuffle(data)
                        data = data[:limit_dataset]
                        logger.info(f"Limiting [{js_file}] to {len(data)} samples")

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

            self.logger.info(f"Ratio: {dataset_config.get('ratio', None)}, final size: [{len(self)}]")

        def load_from_json(self, file_str: str, img_path_base: str):
            self.logger.info(f"Loading [{file_str}]")
            with open(file_str, 'r', encoding="utf-8") as fin:
                data = orjson.loads(fin.read())["data"]
            for e in data:
                e["path"] = os.path.join(img_path_base, e["id"])
            return data

        def __getitem__(self, idx):
            d = self.data[idx]
            img_path = d["path"]
            if not os.path.exists(img_path):
                self.logger.error(f"Image file does not exist: {img_path}")
                return None
            with open(img_path, "rb") as fin:
                buf = fin.read()
            data = {"image": buf, "label": d["gt"], "img_path": d["id"], "ext_data": []}
            return self.transform(data, self.ops)

        def __len__(self):
            return len(self.data)
