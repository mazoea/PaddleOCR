import os
import subprocess
import datetime
import argparse
import logging

env= {
    "log_file": "__maz/__logs/eval_results.log",
    "test_datasets": [
        "eval.ICDAR2013.yml",
        "eval.IBSimple.yml",
        "eval.Invoices.yml",
        "eval.funSD.yml",
        "eval.xFund.yml",
        "eval.IBedits.yml",
        "eval.cord-v2.yml",
        "eval.genText.yml",
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


def run_eval(model, config):
    cmd = [
        "python", "tools/eval.py",
        "-c", f"./__maz/eval/{config}",
        "-o", f"Global.checkpoints={model}"
    ]
    try:
        env = os.environ.copy()
        env["GLOG_minloglevel"] = "2"

        result = subprocess.run(cmd, capture_output=True, text=True, env=env)

        logging.warning(f"Config: {config} | Model: {model}")
        logging.debug(result.stdout)
        logging.debug(result.stderr)

        for line in result.stdout.splitlines():
            if "acc:" in line.lower():
                logging.warning(f"[{config:20s} | {model:60s}] {line}")

    except Exception as e:
        logging.error(f"Error running eval for {config} and {model}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run evaluation with specified models.")
    parser.add_argument('--model1', type=str,
                        help='Path to second model (MODEL1)', default="./__maz/en_PP-OCRv3_rec_train/best_accuracy")
    parser.add_argument('--model2', type=str, required=True, help='Path to second model (MODEL2)')
    args = parser.parse_args()
    _logger.warning(f"Starting evaluation with: {args}")

    for config in env["test_datasets"]:
        logging.warning(f"====")
        run_eval(args.model1, config)
        run_eval(args.model2, config)

if __name__ == "__main__":
    main()
