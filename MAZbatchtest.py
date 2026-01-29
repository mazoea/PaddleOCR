"""
| Dataset          | v3_en_mobile_2024.04.02 | en_PP-OCRv5_mobile_rec |
|------------------|-----------------------:|------------------------:|
| DOCS-BIO         | 0.9470                 | 0.9657                  |
| ICDAR2013        | 0.9375                 | 0.9518                  |
| IBSimple         | 0.9894                 | 0.9676                  |
| IBedits          | 0.9719                 | 0.8631                  |
| IBdiffs          | 0.0                    | 0.5895                  |
| Invoices         | 0.9941                 | 0.9859                  |
| funSD            | 0.9761                 | 0.9642                  |
| xFund            | 0.9882                 | 0.9792                  |
| cord-v2          | 0.9588                 | 0.9526                  |
| genText          | 0.9726                 | 0.5890                  |
| hiertext         | 0.7169                 | 0.8795                  |
| zebra-lines-lite | 0.0                    | 0.5322                  |


Notes:
    hiertext - ratio=[0., 0.25], bigger ratio has +- the same acc
"""

import os
import re
import subprocess
import argparse
import logging
import time

env= {
    "log_file": "__maz/__logs/eval_results.log",
    "test_datasets": [
        "eval.DOCS-BIO.yml",
        "eval.ICDAR2013.yml",
        "eval.IBSimple.yml",
        "eval.IBedits.yml",
        "eval.IBdiffs.yml",
        "eval.Invoices.yml",
        "eval.funSD.yml",
        "eval.xFund.yml",
        "eval.cord-v2.yml",
        "eval.genText.yml",
        "eval.hiertext.yml",
        "eval.zebra-lines-lite.yml",
    ]
}

def setup_logger(log_file: str):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        filename=log_file,
        filemode='w',
        level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s',
    )
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

setup_logger(env["log_file"])
_logger = logging.getLogger()
_logger.warning(f"Logging to {env['log_file']}")


def parse_table_from_comment(script_path: str):
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Match the table and extract all lines
    match = re.search(r'"""\s*(\| Dataset\s*\|.*?\n\|[-|:\s]+\n(?:\|.*\n)+?)"""', content, re.DOTALL)
    if not match:
        raise ValueError("Could not find the markdown table in the comment.")

    lines = match.group(1).strip().splitlines()
    headers = [cell.strip() for cell in lines[0].strip().split("|")[1:-1]]

    rows = {}
    for line in lines[2:]:  # Skip header and separator lines
        cells = [cell.strip() for cell in line.strip().split("|")[1:-1]]
        if len(cells) >= 3:
            dataset = cells[0]
            scores = cells[1:]
            rows[dataset] = [float(x) for x in scores]

    return {
        "header": headers,
        "acc": rows
    }


def run_eval(model, config):
    cmd = [
        "python", "tools/eval.py",
        "-c", f"./__maz/eval/{config}",
        "-o", f"Global.checkpoints={model}",
        #"Global.use_gpu=false"
    ]
    try:
        env = os.environ.copy()
        env["GLOG_minloglevel"] = "2"

        # print(" ".join(cmd))

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            encoding="utf-8",
        )

        # logging.warning(f"Config: {config} | Model: {model}")
        logging.debug(result.stdout)
        logging.debug(result.stderr)

        for line in result.stdout.splitlines():
            if "acc:" in line.lower():
                match = re.search(r'acc:\s*([0-9.]+)', line.lower())
                if match:
                    return float(match.group(1))

    except Exception as e:
        logging.error(f"Error running eval for {config} and {model}: {e}")

    return -100.0


def get_res_tag(orig_score, model1_acc, model2_acc) -> str:
    if "?" in (orig_score, model1_acc, model2_acc):
        return "MISSING"

    try:
        o = round(float(orig_score), 2)
        m1 = round(float(model1_acc), 2)
        m2 = round(float(model2_acc), 2)

        if m2 > max(o, m1):
            result = "BEST"
        elif m2 < min(o, m1):
            result = "WORST"
        elif m2 == m1:
            result = "EQUAL-M1"
        elif m2 == o:
            result = "EQUAL-BASE"
        elif m2 > o:
            result = "IMPROVED-over-BASE"
        elif m2 > m1:
            result = "IMPROVED-over-M1"
        else:
            result = "WORSE"
    except Exception:
        result = "?"

    return result


def main():
    parser = argparse.ArgumentParser(description="Run evaluation with specified models.")
    parser.add_argument('--model1', type=str,
                        help='Path to second model (MODEL1)', default="./__maz/en_PP-OCRv3_rec_train/best_accuracy")
    parser.add_argument('--model2', type=str, required=True, help='Path to second model (MODEL2)')
    args = parser.parse_args()
    _logger.warning(f"Starting evaluation with: {args}")

    baseline_d = parse_table_from_comment(__file__)
    header_msg = "|" .join(f' {x:26s} ' for x in baseline_d["header"])
    _logger.warning(f"|{header_msg}| {args.model2}")

    for config in env["test_datasets"]:
        dataset_name = os.path.basename(config).split(".")[1]
        orig_score, model1_acc = baseline_d["acc"].get(dataset_name, "?")
        # logging.warning(f"====")
        # run_eval(args.model1, config)
        s2 = time.time()
        model2_acc = run_eval(args.model2, config)
        took2 = time.time() - s2

        # based on the other, print out the result BEST, WORST, ...
        result = get_res_tag(orig_score, model1_acc, model2_acc)
        _logger.warning(f"| {dataset_name:26s} | {str(orig_score):26s} | {str(model1_acc):26s} | {model2_acc:6.4f} - {result:20s} | Took {took2:6.2f}s")


if __name__ == "__main__":
    main()
