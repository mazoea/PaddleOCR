import sys
import random
import base64
import os
import glob
import orjson
import tqdm

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

        self.data_dir = dataset_config["data_dir"]
        if isinstance(self.data_dir, str):
            self.data_dir = [self.data_dir]
        self.seed = seed

        self.ops = create_operators(dataset_config["transforms"], global_config)
        self.data = []

        for dt_folder in self.data_dir:
            if os.path.isfile(dt_folder) and dt_folder.endswith(".json"):
                woec_jsons = [dt_folder]
            else:
                woec_jsons = glob.glob(os.path.join(dt_folder, MazDataset.json_glob))
            if len(woec_jsons) == 0:
                logger.critical(f"Missing woec json files in [{dt_folder}]")
                sys.exit(1)
            for woec_json in woec_jsons:
                self.data += self.load_from_json(woec_json)

        random.seed(self.seed)
        random.shuffle(self.data)
        if dataset_config.get("ratio", None) is not None:
            s, e = dataset_config["ratio"]
            s = int(s * len(self.data))
            e = int(e * len(self.data))
            self.data = self.data[s:e]
            self.logger.info(f"Ratio: {dataset_config['ratio']}, final size: [{len(self)}]")


    def load_from_json(self, file_str):
        self.logger.info(f"Loading [{file_str}]")

        with open(file_str, 'r', encoding="utf-8") as fin:
            data = orjson.loads(fin.read())["data"]
        ret = []
        for d in tqdm.tqdm(data):
            gt_score = d["gt"]
            confs = d["confs"]
            gt = d["gttext"]
            if "img_data" not in d:
                raise NotImplementedError("img_data not in json")
            buf = base64.b64decode(d["img_data"])
            row = (gt, buf, gt_score[2])
            ret.append(row)
        return ret

    def __getitem__(self, idx):
        gt, buf, paddle_score = self.data[idx]
        data = {"image": buf, "label": gt, "ext_data": []}
        # in train, operator calls np.frombuffer(img, dtype="uint8")
        outs = self.transform(data, self.ops)
        # e.g., max text length exceeded - select random one
        while outs is None:
            self.logger.info(f"Transform failed for [{gt}]")
            outs = self.__getitem__(random.randint(0, len(self) - 1))
        return outs

    def __len__(self):
        return len(self.data)

