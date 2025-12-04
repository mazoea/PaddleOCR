import argparse
import tqdm
import glob
import shutil
import os
import json

from classification_lines import process
#from cut_out_lines import process

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation on documents sets.")
    parser.add_argument('--doc_sets', type=str, required=True,
                        help='Path to document folder', default="")
    parser.add_argument('--is_zebra', action='store_true', default=False, help='is path zebra path')
    parser.add_argument('--false_positive', type=str, default="", required=True, help='file to log false positive cases')
    parser.add_argument('--false_negative', type=str, default="", required=True, help='file to log false negative cases')
    args = parser.parse_args()

    cnt_correct = 0
    cnt_wrong = 0
    cnt_total = 0
    error = 0

    dataset = []
    dataset_path = "d:/tmp/dataset-lines/"
    # go through png files in th folder
    files = glob.glob(f"{args.doc_sets}/*.png")
    for f in tqdm.tqdm(files):
        if "detection" in f:
            continue
        if "lines_viz" in f:
            continue
        input_bbs = f"{f}_detection.png.bbs.json"

        # Check if input file exists
        if not os.path.exists(f) and not os.path.exists(input_bbs):
            print(f"Error: Input files '{f}' does not exist!")
            continue
        #res, dataset_doc = process(f, input_bbs, False, dataset_path)
        #dataset.extend(dataset_doc)
        res = process(f, input_bbs, False, dataset_path)
        cnt_total += 1
        if res["zebra_doc"]:
            if args.is_zebra:
                cnt_correct += 1
            else:
                cnt_wrong += 1
                if args.false_positive:
                    try:
                        shutil.copy(f, f"{args.false_positive}")
                    except Exception as e:
                        print(f"Error copying file {f} to {args.false_positive}: {e}")
        else:
            if not args.is_zebra:
                cnt_correct += 1
            else:
                cnt_wrong += 1
                if args.false_negative:
                    try:
                        shutil.copy(f, f"{args.false_negative}")
                    except Exception as e:
                        print(f"Error copying file {f} to {args.false_negative}: {e}")

    print(f"Total: {cnt_total}, Correct: {cnt_correct}, Wrong: {cnt_wrong}, Error: {error}")
    accuracy = cnt_correct / cnt_total if cnt_total > 0 else 0.0
    print(f"Accuracy: {accuracy:.4f}")

    #json.dump(dataset, open(f"{dataset_path}/dataset_lines.json", "w", encoding="utf-8"), indent=2)
