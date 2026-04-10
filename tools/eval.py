# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import os
import sys
import json
from collections import defaultdict
from pathlib import Path

__dir__ = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, __dir__)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

import paddle
from ppocr.data import build_dataloader, set_signal_handlers
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.metrics import build_metric
from ppocr.utils.save_load import load_model
import tools.program as program


def write_eval_diffs(metric: dict, config: dict, checkpoint: str | None = None) -> None:
    """Write ``eval.diffs.*.json`` for the current evaluation run.

    Expects ``metric["diffs"]`` to contain ``(pred, target, pred_conf)``
    tuples.  After writing, ``diffs`` and ``diffs-count`` keys are removed
    from *metric* so callers can continue using it.

    *checkpoint* is stored in the JSON; when ``None`` it falls back to
    ``config.get("checkpoints")``.  The checkpoint name used in the
    filename is derived from the path (second segment of
    ``"dir/name/file"`` or the last path component).
    """
    if "diffs" not in metric:
        return

    dataset_name = os.path.basename(
        config["Eval"]["dataset"]["data_dir"]
    ).replace(".PAD.FILTERED.json", "")

    ckpt = checkpoint or config.get("checkpoints", "")
    parts = ckpt.replace("\\", "/").split("/")
    checkpoint_name = parts[1] if len(parts) == 3 else Path(ckpt).name or "UNK"

    cnter: dict[str, int] = defaultdict(int)
    for pred, target, pred_conf in metric["diffs"]:
        if pred != target:
            conf = float(pred_conf) if not isinstance(pred_conf, float) else pred_conf
            k = f"{pred} -> {target} [{conf:5.2f}]"
            cnter[k] += 1

    # Ensure diffs contain only JSON-serializable types (numpy floats → float).
    metric["diffs"] = [(p, t, float(c)) for p, t, c in metric["diffs"]]
    metric["diffs-count"] = sorted(cnter.items(), key=lambda x: x[1], reverse=True)
    metric["checkpoint"] = ckpt

    output_dir = os.environ.get("EVAL_OUTPUT_DIR", "inference_results")
    eval_file = os.path.join(
        output_dir, f"eval.diffs.{checkpoint_name}__{dataset_name}.json"
    )
    os.makedirs(os.path.dirname(eval_file), exist_ok=True)
    with open(eval_file, "w", encoding="utf-8") as fout:
        json.dump(metric, fout, indent=4, ensure_ascii=True)

    del metric["checkpoint"]
    del metric["diffs"]
    del metric["diffs-count"]


def main():
    global_config = config["Global"]
    # build dataloader
    set_signal_handlers()
    valid_dataloader = build_dataloader(config, "Eval", device, logger)

    # build post process
    post_process_class = build_post_process(config["PostProcess"], global_config)

    # build model
    # for rec algorithm
    if hasattr(post_process_class, "character"):
        char_num = len(getattr(post_process_class, "character"))
        if config["Architecture"]["algorithm"] in [
            "Distillation",
        ]:  # distillation model
            for key in config["Architecture"]["Models"]:
                if (
                    config["Architecture"]["Models"][key]["Head"]["name"] == "MultiHead"
                ):  # for multi head
                    out_channels_list = {}
                    if config["PostProcess"]["name"] == "DistillationSARLabelDecode":
                        char_num = char_num - 2
                    if config["PostProcess"]["name"] == "DistillationNRTRLabelDecode":
                        char_num = char_num - 3
                    out_channels_list["CTCLabelDecode"] = char_num
                    out_channels_list["SARLabelDecode"] = char_num + 2
                    out_channels_list["NRTRLabelDecode"] = char_num + 3
                    config["Architecture"]["Models"][key]["Head"][
                        "out_channels_list"
                    ] = out_channels_list
                else:
                    config["Architecture"]["Models"][key]["Head"][
                        "out_channels"
                    ] = char_num
        elif config["Architecture"]["Head"]["name"] == "MultiHead":  # for multi head
            out_channels_list = {}
            if config["PostProcess"]["name"] == "SARLabelDecode":
                char_num = char_num - 2
            if config["PostProcess"]["name"] == "NRTRLabelDecode":
                char_num = char_num - 3
            out_channels_list["CTCLabelDecode"] = char_num
            out_channels_list["SARLabelDecode"] = char_num + 2
            out_channels_list["NRTRLabelDecode"] = char_num + 3
            config["Architecture"]["Head"]["out_channels_list"] = out_channels_list
        else:  # base rec model
            config["Architecture"]["Head"]["out_channels"] = char_num

    model = build_model(config["Architecture"])
    extra_input_models = [
        "SRN",
        "NRTR",
        "SAR",
        "SEED",
        "SVTR",
        "SVTR_LCNet",
        "VisionLAN",
        "RobustScanner",
        "SVTR_HGNet",
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
    if "model_type" in config["Architecture"].keys():
        if config["Architecture"]["algorithm"] == "CAN":
            model_type = "can"
        elif config["Architecture"]["algorithm"] == "LaTeXOCR":
            model_type = "latexocr"
            config["Metric"]["cal_bleu_score"] = True
        elif config["Architecture"]["algorithm"] == "UniMERNet":
            model_type = "unimernet"
            config["Metric"]["cal_bleu_score"] = True
        elif config["Architecture"]["algorithm"] in [
            "PP-FormulaNet-S",
            "PP-FormulaNet-L",
            "PP-FormulaNet_plus-S",
            "PP-FormulaNet_plus-M",
            "PP-FormulaNet_plus-L",
        ]:
            model_type = "pp_formulanet"
            config["Metric"]["cal_bleu_score"] = True
        else:
            model_type = config["Architecture"]["model_type"]
    else:
        model_type = None

    # build metric
    eval_class = build_metric(config["Metric"])
    # amp
    use_amp = config["Global"].get("use_amp", False)
    amp_level = config["Global"].get("amp_level", "O2")
    amp_custom_black_list = config["Global"].get("amp_custom_black_list", [])
    if use_amp:
        AMP_RELATED_FLAGS_SETTING = {
            "FLAGS_cudnn_batchnorm_spatial_persistent": 1,
        }
        paddle.set_flags(AMP_RELATED_FLAGS_SETTING)
        scale_loss = config["Global"].get("scale_loss", 1.0)
        use_dynamic_loss_scaling = config["Global"].get(
            "use_dynamic_loss_scaling", False
        )
        scaler = paddle.amp.GradScaler(
            init_loss_scaling=scale_loss,
            use_dynamic_loss_scaling=use_dynamic_loss_scaling,
        )
        if amp_level == "O2":
            model = paddle.amp.decorate(
                models=model, level=amp_level, master_weight=True
            )
    else:
        scaler = None

    best_model_dict = load_model(
        config, model, model_type=config["Architecture"]["model_type"]
    )
    if len(best_model_dict) and config["Eval"].get("silent", True) == False:
        logger.info("metric in ckpt ***************")
        for k, v in best_model_dict.items():
            logger.info("{}:{}".format(k, v))

    # start eval
    metric = program.eval(
        model,
        valid_dataloader,
        post_process_class,
        eval_class,
        model_type,
        extra_input,
        scaler,
        amp_level,
        amp_custom_black_list,
        silent=os.environ.get("EVAL_VERBOSE", "0") == "0",
    )

    write_eval_diffs(metric, config)

    logger.critical(f"metric eval *************** {config['Eval']['dataset']['data_dir']}")
    for k, v in metric.items():
        msg = "{}:{}".format(k, v)
        logger.info(msg) if k != "acc" else logger.critical(msg)


if __name__ == "__main__":
    config, device, logger, vdl_writer = program.preprocess()
    main()
