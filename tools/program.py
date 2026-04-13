# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import gc
import sys
import platform
import yaml
import time
import datetime
import paddle
import paddle.distributed as dist
from tqdm import tqdm
import cv2
import numpy as np
import copy
from argparse import ArgumentParser, RawDescriptionHelpFormatter

from ppocr.utils.stats import TrainingStats
from ppocr.utils.save_load import save_model
from ppocr.utils.utility import print_dict, AverageMeter
from ppocr.utils.logging import get_logger
from ppocr.utils.loggers import WandbLogger, Loggers
from ppocr.utils import profiler
from ppocr.data import build_dataloader
from ppocr.utils.export_model import export


class ArgsParser(ArgumentParser):
    def __init__(self):
        super(ArgsParser, self).__init__(formatter_class=RawDescriptionHelpFormatter)
        self.add_argument("-c", "--config", help="configuration file to use")
        self.add_argument("-o", "--opt", nargs="+", help="set configuration options")
        self.add_argument(
            "-p",
            "--profiler_options",
            type=str,
            default=None,
            help="The option of profiler, which should be in format "
            '"key1=value1;key2=value2;key3=value3".',
        )

    def parse_args(self, argv=None):
        args = super(ArgsParser, self).parse_args(argv)
        assert args.config is not None, "Please specify --config=configure_file_path."
        args.opt = self._parse_opt(args.opt)
        return args

    def _parse_opt(self, opts):
        config = {}
        if not opts:
            return config
        for s in opts:
            s = s.strip()
            k, v = s.split("=")
            config[k] = yaml.load(v, Loader=yaml.Loader)
        return config


def _expand_env_vars(obj):
    """
    Recursively expand ${VAR} environment variables in all string values.
    Exits with an error if any variable cannot be resolved.
    """
    import re

    _env_pattern = re.compile(r"\$\{([^}]+)\}")

    def _expand_string(s):
        def _replace(m):
            var = m.group(1)
            val = os.environ.get(var)
            if val is None:
                print(
                    "\n"
                    + "=" * 70
                    + f"\n  ERROR: Environment variable '${{{var}}}' is not set!\n"
                    + "  Please set it before running, e.g.:\n"
                    + f"      set {var}=<value>\n"
                    + "=" * 70
                    + "\n"
                )
                sys.exit(1)
            return val

        return _env_pattern.sub(_replace, s)

    if isinstance(obj, str):
        return _expand_string(obj)
    elif isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_expand_env_vars(item) for item in obj]
    return obj


def load_config(file_path):
    """
    Load config from yml/yaml file.
    Args:
        file_path (str): Path of the config file to be loaded.
    Returns: global config
    """
    _, ext = os.path.splitext(file_path)
    assert ext in [".yml", ".yaml"], "only support yaml files for now"
    config = yaml.load(open(file_path, "rb"), Loader=yaml.Loader)
    config = _expand_env_vars(config)
    return config


def merge_config(config, opts):
    """
    Merge config into global config.
    Args:
        config (dict): Config to be merged.
    Returns: global config
    """
    for key, value in opts.items():
        if "." not in key:
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value
        else:
            sub_keys = key.split(".")
            assert sub_keys[0] in config, (
                "the sub_keys can only be one of global_config: {}, but get: "
                "{}, please check your running command".format(
                    config.keys(), sub_keys[0]
                )
            )
            cur = config[sub_keys[0]]
            for idx, sub_key in enumerate(sub_keys[1:]):
                if idx == len(sub_keys) - 2:
                    cur[sub_key] = value
                else:
                    cur = cur[sub_key]
    return config


def check_device(use_gpu, use_xpu=False, use_npu=False, use_mlu=False, use_gcu=False):
    """
    Log error and exit when set use_gpu=true in paddlepaddle
    cpu version.
    """
    err = (
        "Config {} cannot be set as true while your paddle "
        "is not compiled with {} ! \nPlease try: \n"
        "\t1. Install paddlepaddle to run model on {} \n"
        "\t2. Set {} as false in config file to run "
        "model on CPU"
    )

    try:
        if use_gpu and use_xpu:
            print("use_xpu and use_gpu can not both be true.")
        if use_gpu and not paddle.is_compiled_with_cuda():
            print(err.format("use_gpu", "cuda", "gpu", "use_gpu"))
            sys.exit(1)
        if use_xpu and not paddle.device.is_compiled_with_xpu():
            print(err.format("use_xpu", "xpu", "xpu", "use_xpu"))
            sys.exit(1)
        if use_npu:
            if (
                int(paddle.version.major) != 0
                and int(paddle.version.major) <= 2
                and int(paddle.version.minor) <= 4
            ):
                if not paddle.device.is_compiled_with_npu():
                    print(err.format("use_npu", "npu", "npu", "use_npu"))
                    sys.exit(1)
            # is_compiled_with_npu() has been updated after paddle-2.4
            else:
                if not paddle.device.is_compiled_with_custom_device("npu"):
                    print(err.format("use_npu", "npu", "npu", "use_npu"))
                    sys.exit(1)
        if use_mlu and not paddle.device.is_compiled_with_mlu():
            print(err.format("use_mlu", "mlu", "mlu", "use_mlu"))
            sys.exit(1)
        if use_gcu and not paddle.device.is_compiled_with_custom_device("gcu"):
            print(err.format("use_gcu", "gcu", "gcu", "use_gcu"))
            sys.exit(1)
    except Exception as e:
        pass


def to_float32(preds):
    if isinstance(preds, dict):
        for k in preds:
            if isinstance(preds[k], dict) or isinstance(preds[k], list):
                preds[k] = to_float32(preds[k])
            elif isinstance(preds[k], paddle.Tensor):
                preds[k] = preds[k].astype(paddle.float32)
    elif isinstance(preds, list):
        for k in range(len(preds)):
            if isinstance(preds[k], dict):
                preds[k] = to_float32(preds[k])
            elif isinstance(preds[k], list):
                preds[k] = to_float32(preds[k])
            elif isinstance(preds[k], paddle.Tensor):
                preds[k] = preds[k].astype(paddle.float32)
    elif isinstance(preds, paddle.Tensor):
        preds = preds.astype(paddle.float32)
    return preds


def train(
    config,
    train_dataloader,
    valid_dataloader,
    device,
    model,
    loss_class,
    optimizer,
    lr_scheduler,
    post_process_class,
    eval_class,
    pre_best_model_dict,
    logger,
    step_pre_epoch,
    log_writer=None,
    scaler=None,
    amp_level="O2",
    amp_custom_black_list=[],
    amp_custom_white_list=[],
    amp_dtype="float16",
):
    cal_metric_during_train = config["Global"].get("cal_metric_during_train", False)
    calc_epoch_interval = config["Global"].get("calc_epoch_interval", 1)
    log_smooth_window = config["Global"]["log_smooth_window"]
    epoch_num = config["Global"]["epoch_num"]
    print_batch_step = config["Global"]["print_batch_step"]
    eval_batch_step = config["Global"]["eval_batch_step"]
    eval_batch_epoch = config["Global"].get("eval_batch_epoch", None)
    profiler_options = config["profiler_options"]
    print_mem_info = config["Global"].get("print_mem_info", True)
    uniform_output_enabled = config["Global"].get("uniform_output_enabled", False)

    global_step = 0
    if "global_step" in pre_best_model_dict:
        global_step = pre_best_model_dict["global_step"]
    start_eval_step = 0
    if isinstance(eval_batch_step, list) and len(eval_batch_step) >= 2:
        start_eval_step = eval_batch_step[0] if not eval_batch_epoch else 0
        eval_batch_step = (
            eval_batch_step[1]
            if not eval_batch_epoch
            else step_pre_epoch * eval_batch_epoch
        )
        if len(valid_dataloader or []) == 0:
            logger.info(
                "No Images in eval dataset, evaluation during training "
                "will be disabled"
            )
            start_eval_step = 1e111
        logger.info(
            "During the training process, after the {}th iteration, "
            "an evaluation is run every {} iterations".format(
                start_eval_step, eval_batch_step
            )
        )
    save_epoch_step = config["Global"]["save_epoch_step"]
    save_model_dir = config["Global"]["save_model_dir"]
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    main_indicator = eval_class.main_indicator
    best_model_dict = {main_indicator: 0}
    best_model_dict.update(pre_best_model_dict)
    train_stats = TrainingStats(log_smooth_window, ["lr"])
    model_average = False
    model.train()

    use_srn = config["Architecture"]["algorithm"] == "SRN"
    extra_input_models = [
        "SRN",
        "NRTR",
        "SAR",
        "SEED",
        "SVTR",
        "SVTR_LCNet",
        "SPIN",
        "VisionLAN",
        "RobustScanner",
        "RFL",
        "DRRG",
        "SATRN",
        "SVTR_HGNet",
        "ParseQ",
        "CPPD",
    ]
    extra_input = False
    if config["Architecture"]["algorithm"] == "Distillation":
        for key in config["Architecture"]["Models"]:
            extra_input = (
                extra_input
                or config["Architecture"]["Models"][key]["algorithm"]
                in extra_input_models
            )
    else:
        extra_input = config["Architecture"]["algorithm"] in extra_input_models
    try:
        model_type = config["Architecture"]["model_type"]
    except:
        model_type = None

    algorithm = config["Architecture"]["algorithm"]

    start_epoch = (
        best_model_dict["start_epoch"] if "start_epoch" in best_model_dict else 1
    )

    total_samples = 0
    train_reader_cost = 0.0
    train_batch_cost = 0.0
    reader_start = time.time()
    eta_meter = AverageMeter()

    max_iter = (
        len(train_dataloader) - 1
        if platform.system() == "Windows"
        else len(train_dataloader)
    )

    dbg = os.getenv("MAZOEA_DBG_TRAINING_INPUT", None)

    for epoch in range(start_epoch, epoch_num + 1):
        if train_dataloader.dataset.need_reset:
            train_dataloader = build_dataloader(
                config, "Train", device, logger, seed=epoch
            )
            max_iter = (
                len(train_dataloader) - 1
                if platform.system() == "Windows"
                else len(train_dataloader)
            )

        dbg_max_per_epoch = 250
        dbg_saved_this_epoch = 0
        dbg_saved_train_this_epoch = 0
        dbg_dir = None
        if dbg and dist.get_rank() == 0:
            dbg_dir = os.path.join(
                save_model_dir,
                f"debug_epoch_{epoch}",
            )
            os.makedirs(dbg_dir, exist_ok=True)

        def dbg_dump(images_cpu, prefix, name_suffixes=None):
            """
            Fixed version of dbg_dump that correctly handles visualization of training data.

            Args:
                images_cpu: numpy array of images/masks
                    - 4D: (N, C, H, W) - batch of multi-channel images
                    - 3D: (N, H, W) - batch of single-channel images/masks
                    - 2D: (H, W) - single image/mask
                prefix: string prefix for output filenames
            """
            rand = np.random.randint(0, 10000)

            def normalize_to_uint8(img):
                """Normalize image to [0, 255] range using actual min/max values."""
                img_min = img.min()
                img_max = img.max()
                if img_max > img_min:
                    img = (img - img_min) / (img_max - img_min) * 255
                else:
                    img = np.zeros_like(img)
                return img.astype(np.uint8)

            if images_cpu.ndim == 4:  # Batch of multi-channel images (N, C, H, W)
                for i, img in enumerate(images_cpu):
                    # Convert from (C, H, W) to (H, W, C)
                    img = np.transpose(img, (1, 2, 0))

                    # Normalize to uint8
                    img = normalize_to_uint8(img)

                    # Handle different channel counts
                    if img.shape[2] == 1:
                        # Single channel - remove channel dimension
                        img = img.squeeze(2)
                    elif img.shape[2] == 3:
                        # RGB image - convert to BGR for OpenCV
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    elif img.shape[2] == 2:
                        # Two channels - save first channel (e.g., threshold map)
                        img = img[:, :, 0]
                    else:
                        # More than 3 channels - save first channel
                        img = img[:, :, 0]

                    # Save image
                    suffix = ""
                    if name_suffixes is not None and i < len(name_suffixes) and name_suffixes[i]:
                        suffix = f"_{name_suffixes[i]}"
                    cv2.imwrite(os.path.join(dbg_dir, f'{prefix}_{str(rand)}_{i}{suffix}.png'), img)

            elif images_cpu.ndim == 3:  # Batch of single-channel images (N, H, W)
                for i, img in enumerate(images_cpu):
                    # BUG FIX: Use `img` not `images_cpu`
                    # Original: img = np.transpose(images_cpu, (1, 2, 0))
                    # Fixed: img is already (H, W), no transpose needed

                    # Normalize to uint8
                    img = normalize_to_uint8(img)

                    # Save image (already in correct format for grayscale)
                    suffix = ""
                    if name_suffixes is not None and i < len(name_suffixes) and name_suffixes[i]:
                        suffix = f"_{name_suffixes[i]}"
                    cv2.imwrite(os.path.join(dbg_dir, f'{prefix}_{str(rand)}_{i}{suffix}.png'), img)

            elif images_cpu.ndim == 2:  # Single image (H, W)
                img = normalize_to_uint8(images_cpu)
                cv2.imwrite(os.path.join(dbg_dir, f'{prefix}_{str(rand)}_0.png'), img)

            else:
                #print(f"Warning: Unsupported tensor shape {images_cpu.shape} for {prefix}")
                pass
            return

        def dbg_encode_label_for_filename(text):
            enc = str(text)
            enc = enc.replace("$", "~DOLLAR~")
            enc = enc.replace("[", "~lbracket~")
            enc = enc.replace("]", "~rbracket~")
            enc = enc.replace(" ", "___")
            enc = "".join(
                ch if (ch.isalnum() or ch in ("-", "_", "~")) else "_"
                for ch in enc
            )
            return f"LABEL_{enc[:80]}"

        def dbg_decode_label_suffixes(labels_batch):
            suffixes = []
            try:
                if not hasattr(post_process_class, "decode"):
                    return suffixes
                if labels_batch is None or len(labels_batch) < 2:
                    return suffixes

                labels = labels_batch[1]
                if isinstance(labels, paddle.Tensor):
                    labels = labels.numpy()

                decoded_labels = post_process_class.decode(labels)
                for item in decoded_labels:
                    if isinstance(item, (tuple, list)) and len(item) >= 1:
                        text = item[0]
                    else:
                        text = str(item)

                    safe_text = dbg_encode_label_for_filename(text)
                    suffixes.append(safe_text)
            except Exception as e:
                logger.info(f"Debug label decode failed for filename suffixes: {e}")
                suffixes = []
            return suffixes

        def dbg_dump_rec(images_cpu, preds_out, labels_batch, n_to_dump):
            if dbg_dir is None or n_to_dump <= 0:
                return 0

            def normalize_to_uint8(img):
                img_min = img.min()
                img_max = img.max()
                if img_max > img_min:
                    img = (img - img_min) / (img_max - img_min) * 255
                else:
                    img = np.zeros_like(img)
                return img.astype(np.uint8)

            # Decode recognition outputs to human-readable text when possible.
            pred_texts = []
            label_texts = []
            label_for_decode = labels_batch[1]
            if isinstance(label_for_decode, paddle.Tensor):
                label_for_decode = label_for_decode.numpy()

            # Decode ground-truth labels independently so records stay useful
            # even when prediction decode fails.
            try:
                if hasattr(post_process_class, "decode"):
                    gt_decoded = post_process_class.decode(label_for_decode)
                    if isinstance(gt_decoded, list):
                        label_texts = gt_decoded
            except Exception as e:
                logger.info(f"Debug label decode failed for records: {e}")

            try:
                if config["Loss"]["name"] in ["MultiLoss", "MultiLoss_v2"] and isinstance(preds_out, dict) and "ctc" in preds_out:
                    decoded = post_process_class(preds_out["ctc"], label_for_decode)
                else:
                    decoded = post_process_class(preds_out, label_for_decode)

                if isinstance(decoded, (tuple, list)) and len(decoded) >= 2:
                    pred_texts = decoded[0] if isinstance(decoded[0], list) else []
                    if not label_texts:
                        label_texts = decoded[1] if isinstance(decoded[1], list) else []
                elif isinstance(decoded, list):
                    pred_texts = decoded
            except Exception as e:
                logger.info(f"Debug decode failed for recognition outputs: {e}")
                # Keep debug dump best-effort even when decoding fails.
                pred_texts = []

            dumped = 0
            batch_size = min(images_cpu.shape[0], n_to_dump)
            for i in range(batch_size):
                img = images_cpu[i]
                if img.ndim == 3:
                    img = np.transpose(img, (1, 2, 0))
                img = normalize_to_uint8(img)
                if img.ndim == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                elif img.ndim == 3 and img.shape[2] == 1:
                    img = img.squeeze(2)

                pred_item = pred_texts[i] if i < len(pred_texts) else None
                label_item = label_texts[i] if i < len(label_texts) else None

                pred_text, pred_score = "", ""
                if isinstance(pred_item, (tuple, list)) and len(pred_item) >= 2:
                    pred_text, pred_score = pred_item[0], pred_item[1]
                elif pred_item is not None:
                    pred_text = str(pred_item)

                label_text = ""
                if isinstance(label_item, (tuple, list)) and len(label_item) >= 1:
                    label_text = label_item[0]
                elif label_item is not None:
                    label_text = str(label_item)

                safe_label = ""
                if label_text:
                    safe_label = dbg_encode_label_for_filename(label_text)

                sample_id = dbg_saved_this_epoch + dumped
                file_name = f"sample_{sample_id:03d}_step_{global_step:06d}"
                if safe_label:
                    file_name = f"{file_name}_{safe_label}"
                image_file = os.path.join(dbg_dir, f"{file_name}.png")
                cv2.imwrite(image_file, img)

                with open(
                    os.path.join(dbg_dir, "records.txt"),
                    "a",
                    encoding="utf-8",
                ) as fout:
                    nospace_eq = pred_text.replace(" ", "") == label_text.replace(" ", "")
                    fout.write(
                        f"step={global_step} match/NOSPACEmatch={str(pred_text==label_text).upper():>5s}/{str(nospace_eq).upper():>5s}    pred=[{pred_text}] label=[{label_text}] score={pred_score}  file={os.path.basename(image_file)}, \n"
                    )
                dumped += 1
            return dumped

        for idx, batch in enumerate(train_dataloader):
            # Keep dumping train inputs across batches until debug quota is reached.
            if (
                dbg
                and dist.get_rank() == 0
                and dbg_saved_train_this_epoch < dbg_max_per_epoch
            ):
                remaining_train = dbg_max_per_epoch - dbg_saved_train_this_epoch
                dbg_imgs = batch[0].cpu().numpy()
                n_dump = min(dbg_imgs.shape[0], remaining_train)
                dbg_name_suffixes = dbg_decode_label_suffixes(batch)
                dbg_dump(
                    dbg_imgs[:n_dump],
                    "train",
                    dbg_name_suffixes[:n_dump] if dbg_name_suffixes else None,
                )
                dbg_saved_train_this_epoch += n_dump

            model.train()
            profiler.add_profiler_step(profiler_options)
            train_reader_cost += time.time() - reader_start
            if idx >= max_iter:
                break
            lr = optimizer.get_lr()
            images = batch[0]
            if use_srn:
                model_average = True
            # use amp
            if scaler:
                with paddle.amp.auto_cast(
                    level=amp_level,
                    custom_black_list=amp_custom_black_list,
                    custom_white_list=amp_custom_white_list,
                    dtype=amp_dtype,
                ):
                    if model_type == "table" or extra_input:
                        preds = model(images, data=batch[1:])
                    elif model_type in ["kie"]:
                        preds = model(batch)
                    elif algorithm in ["CAN"]:
                        preds = model(batch[:3])
                    elif algorithm in [
                        "LaTeXOCR",
                        "UniMERNet",
                        "PP-FormulaNet-S",
                        "PP-FormulaNet-L",
                        "PP-FormulaNet_plus-S",
                        "PP-FormulaNet_plus-M",
                        "PP-FormulaNet_plus-L",
                    ]:
                        preds = model(batch)
                    else:
                        preds = model(images)
                preds = to_float32(preds)

                if dbg and dist.get_rank() == 0 and dbg_saved_this_epoch < dbg_max_per_epoch:
                    remaining = dbg_max_per_epoch - dbg_saved_this_epoch
                    images_cpu = images.cpu().numpy()
                    dbg_saved_this_epoch += dbg_dump_rec(images_cpu, preds, batch, remaining)

                loss = loss_class(preds, batch)
                avg_loss = loss["loss"]
                scaled_avg_loss = scaler.scale(avg_loss)
                scaled_avg_loss.backward()
                scaler.minimize(optimizer, scaled_avg_loss)
            else:
                if model_type == "table" or extra_input:
                    preds = model(images, data=batch[1:])
                elif model_type in ["kie", "sr"]:
                    preds = model(batch)
                elif algorithm in ["CAN"]:
                    preds = model(batch[:3])
                elif algorithm in [
                    "LaTeXOCR",
                    "UniMERNet",
                    "PP-FormulaNet-S",
                    "PP-FormulaNet-L",
                    "PP-FormulaNet_plus-S",
                    "PP-FormulaNet_plus-M",
                    "PP-FormulaNet_plus-L",
                ]:
                    preds = model(batch)
                else:
                    preds = model(images)

                if dbg and dist.get_rank() == 0 and dbg_saved_this_epoch < dbg_max_per_epoch:
                    remaining = dbg_max_per_epoch - dbg_saved_this_epoch
                    images_cpu = images.cpu().numpy()
                    dbg_saved_this_epoch += dbg_dump_rec(images_cpu, preds, batch, remaining)

                # Keep old detection-style debug outputs for compatible branches.
                if dbg and isinstance(preds, dict) and "maps" in preds and len(batch) >= 5:
                    preds_cpu = preds["maps"].cpu().numpy()
                    dbg_dump(preds_cpu, "preds")
                    label_threshold_map = batch[1].cpu().numpy()
                    dbg_dump(label_threshold_map, "label_threshold_map")
                    label_threshold_mask = batch[2].cpu().numpy()
                    dbg_dump(label_threshold_mask, "label_threshold_mask")
                    label_shrink_map = batch[3].cpu().numpy()
                    dbg_dump(label_shrink_map, "label_shrink_map")
                    label_shrink_mask = batch[4].cpu().numpy()
                    dbg_dump(label_shrink_mask, "label_shrink_mask")

                loss = loss_class(preds, batch)
                avg_loss = loss["loss"]
                avg_loss.backward()
                optimizer.step()

            optimizer.clear_grad()

            if (
                cal_metric_during_train and epoch % calc_epoch_interval == 0
            ):  # only rec and cls need
                batch = [item.numpy() for item in batch]
                if model_type in ["kie", "sr"]:
                    eval_class(preds, batch)
                elif model_type in ["table"]:
                    post_result = post_process_class(preds, batch)
                    eval_class(post_result, batch)
                elif algorithm in ["CAN"]:
                    model_type = "can"
                    eval_class(preds[0], batch[2:], epoch_reset=(idx == 0))
                elif algorithm in ["LaTeXOCR"]:
                    model_type = "latexocr"
                    post_result = post_process_class(preds, batch[1], mode="train")
                    eval_class(post_result[0], post_result[1], epoch_reset=(idx == 0))
                elif algorithm in ["UniMERNet"]:
                    model_type = "unimernet"
                    post_result = post_process_class(preds[0], batch[1], mode="train")
                    eval_class(post_result[0], post_result[1], epoch_reset=(idx == 0))
                elif algorithm in [
                    "PP-FormulaNet-S",
                    "PP-FormulaNet-L",
                    "PP-FormulaNet_plus-S",
                    "PP-FormulaNet_plus-M",
                    "PP-FormulaNet_plus-L",
                ]:
                    model_type = "pp_formulanet"
                    post_result = post_process_class(preds[0], batch[1], mode="train")
                    eval_class(post_result[0], post_result[1], epoch_reset=(idx == 0))
                else:
                    if config["Loss"]["name"] in [
                        "MultiLoss",
                        "MultiLoss_v2",
                    ]:  # for multi head loss
                        post_result = post_process_class(
                            preds["ctc"], batch[1]
                        )  # for CTC head out
                    elif config["Loss"]["name"] in ["VLLoss"]:
                        post_result = post_process_class(preds, batch[1], batch[-1])
                    else:
                        post_result = post_process_class(preds, batch[1])
                    eval_class(post_result, batch)
                metric = eval_class.get_metric()
                train_stats.update(metric)

            train_batch_time = time.time() - reader_start
            train_batch_cost += train_batch_time
            eta_meter.update(train_batch_time)
            global_step += 1
            total_samples += len(images)

            if not isinstance(lr_scheduler, float):
                lr_scheduler.step()

            # logger and visualdl
            stats = {
                k: float(v) if v.shape == [] else v.numpy().mean()
                for k, v in loss.items()
            }
            stats["lr"] = lr
            train_stats.update(stats)

            if log_writer is not None and dist.get_rank() == 0:
                log_writer.log_metrics(
                    metrics=train_stats.get(), prefix="TRAIN", step=global_step
                )

            if (global_step > 0 and global_step % print_batch_step == 0) or (
                idx >= len(train_dataloader) - 1
            ):
                logs = train_stats.log()

                eta_sec = (
                    (epoch_num + 1 - epoch) * len(train_dataloader) - idx - 1
                ) * eta_meter.avg
                eta_sec_format = str(datetime.timedelta(seconds=int(eta_sec)))
                max_mem_reserved_str = ""
                max_mem_allocated_str = ""
                if paddle.device.is_compiled_with_cuda() and print_mem_info:
                    max_mem_reserved_str = f", max_mem_reserved: {paddle.device.cuda.max_memory_reserved() // (1024 ** 2)} MB,"
                    max_mem_allocated_str = f" max_mem_allocated: {paddle.device.cuda.max_memory_allocated() // (1024 ** 2)} MB"
                strs = (
                    "epoch: [{}/{}], global_step: {}, {}, avg_reader_cost: "
                    "{:.5f} s, avg_batch_cost: {:.5f} s, avg_samples: {}, "
                    "ips: {:.5f} samples/s, eta: {}{}{}".format(
                        epoch,
                        epoch_num,
                        global_step,
                        logs,
                        train_reader_cost / print_batch_step,
                        train_batch_cost / print_batch_step,
                        total_samples / print_batch_step,
                        total_samples / train_batch_cost,
                        eta_sec_format,
                        max_mem_reserved_str,
                        max_mem_allocated_str,
                    )
                )
                logger.info(strs)

                total_samples = 0
                train_reader_cost = 0.0
                train_batch_cost = 0.0
            # eval
            if (
                global_step > start_eval_step
                and (global_step - start_eval_step) % eval_batch_step == 0
                and dist.get_rank() == 0
            ):
                if model_average:
                    Model_Average = paddle.incubate.ModelAverage(
                        0.15,
                        parameters=model.parameters(),
                        min_average_window=10000,
                        max_average_window=15625,
                    )
                    Model_Average.apply()
                cur_metric = eval(
                    model,
                    valid_dataloader,
                    post_process_class,
                    eval_class,
                    model_type,
                    extra_input=extra_input,
                    scaler=scaler,
                    amp_level=amp_level,
                    amp_custom_black_list=amp_custom_black_list,
                    amp_custom_white_list=amp_custom_white_list,
                    amp_dtype=amp_dtype,
                    silent=True,
                    return_diffs=False,
                )
                cur_metric_str = "cur metric, {}".format(
                    ", ".join(["{}: {}".format(k, v) for k, v in cur_metric.items()])
                )
                logger.info(cur_metric_str)

                # logger metric
                if log_writer is not None:
                    log_writer.log_metrics(
                        metrics=cur_metric, prefix="EVAL", step=global_step
                    )

                if cur_metric[main_indicator] >= best_model_dict[main_indicator]:
                    best_model_dict.update(cur_metric)
                    best_model_dict["best_epoch"] = epoch
                    prefix = "best_accuracy"
                    if uniform_output_enabled:
                        export(
                            config,
                            model,
                            os.path.join(save_model_dir, prefix, "inference"),
                        )
                        gc.collect()
                        model_info = {"epoch": epoch, "metric": best_model_dict}
                    else:
                        model_info = None
                    save_model(
                        model,
                        optimizer,
                        (
                            os.path.join(save_model_dir, prefix)
                            if uniform_output_enabled
                            else save_model_dir
                        ),
                        logger,
                        config,
                        is_best=True,
                        prefix=prefix,
                        save_model_info=model_info,
                        best_model_dict=best_model_dict,
                        epoch=epoch,
                        global_step=global_step,
                        metric=cur_metric[main_indicator],
                    )
                best_str = "best metric, {}".format(
                    ", ".join(
                        ["{}: {}".format(k, v) for k, v in best_model_dict.items()]
                    )
                )
                logger.info(best_str)
                # logger best metric
                if log_writer is not None:
                    log_writer.log_metrics(
                        metrics={
                            "best_{}".format(main_indicator): best_model_dict[
                                main_indicator
                            ]
                        },
                        prefix="EVAL",
                        step=global_step,
                    )

                    log_writer.log_model(
                        is_best=True, prefix="best_accuracy", metadata=best_model_dict
                    )

            reader_start = time.time()
        if dist.get_rank() == 0:
            prefix = "latest"
            if uniform_output_enabled:
                export(config, model, os.path.join(save_model_dir, prefix, "inference"))
                gc.collect()
                model_info = {"epoch": epoch, "metric": best_model_dict}
            else:
                model_info = None
            save_model(
                model,
                optimizer,
                (
                    os.path.join(save_model_dir, prefix)
                    if uniform_output_enabled
                    else save_model_dir
                ),
                logger,
                config,
                is_best=False,
                prefix=prefix,
                save_model_info=model_info,
                best_model_dict=best_model_dict,
                epoch=epoch,
                global_step=global_step,
            )

            if log_writer is not None:
                log_writer.log_model(is_best=False, prefix="latest")

        if dist.get_rank() == 0 and epoch > 0 and epoch % save_epoch_step == 0:
            prefix = "iter_epoch_{}".format(epoch)
            if uniform_output_enabled:
                export(config, model, os.path.join(save_model_dir, prefix, "inference"))
                gc.collect()
                model_info = {"epoch": epoch, "metric": best_model_dict}
            else:
                model_info = None
            save_model(
                model,
                optimizer,
                (
                    os.path.join(save_model_dir, prefix)
                    if uniform_output_enabled
                    else save_model_dir
                ),
                logger,
                config,
                is_best=False,
                prefix=prefix,
                save_model_info=model_info,
                best_model_dict=best_model_dict,
                epoch=epoch,
                global_step=global_step,
                done_flag=epoch == config["Global"]["epoch_num"],
            )
            if log_writer is not None:
                log_writer.log_model(
                    is_best=False, prefix="iter_epoch_{}".format(epoch)
                )

    best_str = "best metric, {}".format(
        ", ".join(["{}: {}".format(k, v) for k, v in best_model_dict.items()])
    )
    logger.info(best_str)
    if dist.get_rank() == 0 and log_writer is not None:
        log_writer.close()
    return


def eval(
    model,
    valid_dataloader,
    post_process_class,
    eval_class,
    model_type=None,
    extra_input=False,
    scaler=None,
    amp_level="O2",
    amp_custom_black_list=[],
    amp_custom_white_list=[],
    amp_dtype="float16",
    silent=False,
    return_diffs=True,
):
    model.eval()
    diffs = []
    with paddle.no_grad():
        total_frame = 0.0
        total_time = 0.0
        pbar = tqdm(
            total=len(valid_dataloader), desc="eval model", position=0, leave=True
        ) if not silent else None
        max_iter = (
            len(valid_dataloader) - 1
            if platform.system() == "Windows"
            and getattr(valid_dataloader, "num_workers", 0) > 0
            else len(valid_dataloader)
        )
        sum_images = 0
        for idx, batch in enumerate(valid_dataloader):
            if idx >= max_iter:
                break
            images = batch[0]
            start = time.time()

            # use amp
            if scaler:
                with paddle.amp.auto_cast(
                    level=amp_level,
                    custom_black_list=amp_custom_black_list,
                    dtype=amp_dtype,
                ):
                    if model_type == "table" or extra_input:
                        preds = model(images, data=batch[1:])
                    elif model_type in ["kie"]:
                        preds = model(batch)
                    elif model_type in ["can"]:
                        preds = model(batch[:3])
                    elif model_type in ["latexocr"]:
                        preds = model(batch)
                    elif model_type in ["sr"]:
                        preds = model(batch)
                        sr_img = preds["sr_img"]
                        lr_img = preds["lr_img"]
                    else:
                        preds = model(images)
                preds = to_float32(preds)
            else:
                if model_type == "table" or extra_input:
                    preds = model(images, data=batch[1:])
                elif model_type in ["kie"]:
                    preds = model(batch)
                elif model_type in ["can"]:
                    preds = model(batch[:3])
                elif model_type in ["latexocr", "unimernet", "pp_formulanet"]:
                    preds = model(batch)
                elif model_type in ["sr"]:
                    preds = model(batch)
                    sr_img = preds["sr_img"]
                    lr_img = preds["lr_img"]
                else:
                    preds = model(images)

            batch_numpy = []
            for item in batch:
                if isinstance(item, paddle.Tensor):
                    batch_numpy.append(item.numpy())
                else:
                    batch_numpy.append(item)
            # Obtain usable results from post-processing methods
            total_time += time.time() - start
            # Evaluate the results of the current batch
            if model_type in ["table", "kie"]:
                if post_process_class is None:
                    eval_class(preds, batch_numpy)
                else:
                    post_result = post_process_class(preds, batch_numpy)
                    eval_class(post_result, batch_numpy)
            elif model_type in ["sr"]:
                eval_class(preds, batch_numpy)
            elif model_type in ["can"]:
                eval_class(preds[0], batch_numpy[2:], epoch_reset=(idx == 0))
            elif model_type in ["latexocr", "unimernet", "pp_formulanet"]:
                post_result = post_process_class(preds, batch[1], "eval")
                eval_class(post_result[0], post_result[1], epoch_reset=(idx == 0))
            else:
                post_result = post_process_class(preds, batch_numpy[1])
                d = eval_class(post_result, batch_numpy)
                if "diffs" in d:
                    diffs += [x for x in d["diffs"] if x[0] != None]

            if pbar:
                pbar.update(1)
            total_frame += len(images)
            sum_images += 1
        # Get final metric，eg. acc or hmean
        metric = eval_class.get_metric()

    if pbar:
        pbar.close()
    model.train()
    # Avoid ZeroDivisionError
    if total_time > 0:
        metric["fps"] = total_frame / total_time
    else:
        metric["fps"] = 0  # or set to a fallback value
    if return_diffs:
        metric["diffs"] = diffs
    return metric


def update_center(char_center, post_result, preds):
    result, label = post_result
    feats, logits = preds
    logits = paddle.argmax(logits, axis=-1)
    feats = feats.numpy()
    logits = logits.numpy()

    for idx_sample in range(len(label)):
        if result[idx_sample][0] == label[idx_sample][0]:
            feat = feats[idx_sample]
            logit = logits[idx_sample]
            for idx_time in range(len(logit)):
                index = logit[idx_time]
                if index in char_center.keys():
                    char_center[index][0] = (
                        char_center[index][0] * char_center[index][1] + feat[idx_time]
                    ) / (char_center[index][1] + 1)
                    char_center[index][1] += 1
                else:
                    char_center[index] = [feat[idx_time], 1]
    return char_center


def get_center(model, eval_dataloader, post_process_class):
    pbar = tqdm(total=len(eval_dataloader), desc="get center:")
    max_iter = (
        len(eval_dataloader) - 1
        if platform.system() == "Windows"
        else len(eval_dataloader)
    )
    char_center = dict()
    for idx, batch in enumerate(eval_dataloader):
        if idx >= max_iter:
            break
        images = batch[0]
        start = time.time()
        preds = model(images)

        batch = [item.numpy() for item in batch]
        # Obtain usable results from post-processing methods
        post_result = post_process_class(preds, batch[1])

        # update char_center
        char_center = update_center(char_center, post_result, preds)
        pbar.update(1)

    pbar.close()
    for key in char_center.keys():
        char_center[key] = char_center[key][0]
    return char_center


def preprocess(is_train=False):
    FLAGS = ArgsParser().parse_args()
    profiler_options = FLAGS.profiler_options
    config = load_config(FLAGS.config)
    config = merge_config(config, FLAGS.opt)
    profile_dic = {"profiler_options": FLAGS.profiler_options}
    config = merge_config(config, profile_dic)

    log_level = logging.DEBUG
    if is_train:
        # save_config
        run_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_model_dir = os.path.join(config["Global"]["save_model_dir"], run_stamp)
        config["Global"]["save_model_dir"] = save_model_dir
        os.makedirs(save_model_dir, exist_ok=True)
        with open(os.path.join(save_model_dir, "config.yml"), "w") as f:
            yaml.dump(dict(config), f, default_flow_style=False, sort_keys=False)
        log_file = "{}/train.log".format(save_model_dir)
        log_level = logging.INFO
    else:
        log_file = None

    log_ranks = config["Global"].get("log_ranks", "0")
    logger = get_logger(log_file=log_file, log_level=log_level, log_ranks=log_ranks)

    # check if set use_gpu=True in paddlepaddle cpu version
    use_gpu = config["Global"].get("use_gpu", False)
    use_xpu = config["Global"].get("use_xpu", False)
    use_npu = config["Global"].get("use_npu", False)
    use_mlu = config["Global"].get("use_mlu", False)
    use_gcu = config["Global"].get("use_gcu", False)

    alg = config["Architecture"]["algorithm"]
    assert alg in [
        "EAST",
        "DB",
        "SAST",
        "Rosetta",
        "CRNN",
        "STARNet",
        "RARE",
        "SRN",
        "CLS",
        "PGNet",
        "Distillation",
        "NRTR",
        "TableAttn",
        "SAR",
        "PSE",
        "SEED",
        "SDMGR",
        "LayoutXLM",
        "LayoutLM",
        "LayoutLMv2",
        "PREN",
        "FCE",
        "SVTR",
        "SVTR_LCNet",
        "ViTSTR",
        "ABINet",
        "DB++",
        "TableMaster",
        "SPIN",
        "VisionLAN",
        "Gestalt",
        "SLANet",
        "RobustScanner",
        "CT",
        "RFL",
        "DRRG",
        "CAN",
        "Telescope",
        "SATRN",
        "SVTR_HGNet",
        "ParseQ",
        "CPPD",
        "LaTeXOCR",
        "UniMERNet",
        "SLANeXt",
        "PP-FormulaNet-S",
        "PP-FormulaNet-L",
        "PP-FormulaNet_plus-S",
        "PP-FormulaNet_plus-M",
        "PP-FormulaNet_plus-L",
    ]

    if use_xpu:
        device = "xpu:{0}".format(os.getenv("FLAGS_selected_xpus", 0))
    elif use_npu:
        device = "npu:{0}".format(os.getenv("FLAGS_selected_npus", 0))
    elif use_mlu:
        device = "mlu:{0}".format(os.getenv("FLAGS_selected_mlus", 0))
    elif use_gcu:  # Use Enflame GCU(General Compute Unit)
        device = "gcu:{0}".format(os.getenv("FLAGS_selected_gcus", 0))
    else:
        device = "gpu:{}".format(dist.ParallelEnv().dev_id) if use_gpu else "cpu"
    check_device(use_gpu, use_xpu, use_npu, use_mlu, use_gcu)

    device = paddle.set_device(device)

    config["Global"]["distributed"] = dist.get_world_size() != 1

    loggers = []

    if "use_visualdl" in config["Global"] and config["Global"]["use_visualdl"]:
        logger.warning(
            "You are using VisualDL, the VisualDL is deprecated and "
            "removed in ppocr!"
        )
        log_writer = None
    if (
        "use_wandb" in config["Global"] and config["Global"]["use_wandb"]
    ) or "wandb" in config:
        save_dir = config["Global"]["save_model_dir"]
        wandb_writer_path = "{}/wandb".format(save_dir)
        if "wandb" in config:
            wandb_params = config["wandb"]
        else:
            wandb_params = dict()
        wandb_params.update({"save_dir": save_dir})
        log_writer = WandbLogger(**wandb_params, config=config)
        loggers.append(log_writer)
    else:
        log_writer = None

    should_print = is_train is True or not config.get("Eval", {}).get("silent", True)
    if should_print:
        print_dict(config, logger)

    if loggers:
        log_writer = Loggers(loggers)
    else:
        log_writer = None

    if should_print:
        logger.info("train with paddle {} and device {}".format(paddle.__version__, device))
    return config, device, logger, log_writer
