import json
import os
import sys
import argparse
import cv2
import tqdm

from coords import bbox

from classification_lines import line_process


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="Classify line dataset results.")
    argparser.add_argument('--input', type=str, required=True, help='Path to input json file')

    args = argparser.parse_args()
    input_path = args.input
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' does not exist!")
        sys.exit(1)

    with open(input_path) as f:
        data_all = json.load(f)

    data = data_all.get("data", [])

    error = 0
    total = 0
    correct = 0
    wrong = 0

    for d in tqdm.tqdm(data):
        id = d.get("id", "")
        gt = d.get("classification", "")
        bbs_d = d.get("bbs", [])

        img_path = os.path.join(os.path.dirname(input_path), id)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        #cv2.imwrite("d:/tmp/gap_roi.png", img)
        if not os.path.exists(img_path):
            print(f"Image file does not exist: {img_path}")
            continue

        bbs = [bbox.default_bbox(b) for b in bbs_d]
        ret, ratio = line_process(img, bbs, False)
        total += 1
        if ret:
            if gt == "zebra":
                correct += 1
            else:
                wrong += 1
                d["classification"] = "zebra"
        else:
            if gt == "classic":
                correct += 1
            else:
                wrong += 1
                d["classification"] = "classic"

    if 0 < total:
        print(f"Correct: {correct}/{total}")
        print(f"Wrong: {wrong}/{total}")
        accuracy = correct / total if total > 0 else 0.0
        print(f"Accuracy: {accuracy:.4f}")
    else:
        print("No valid samples found.")

    #json.dump(data_all, open(f"{input_path}_fixed.json", "w"), indent=2)

