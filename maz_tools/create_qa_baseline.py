"""
    It goes thorugh QA jsons and find words, with confidence > 80 and length > 3
    and store their bounding boxes in a separate json file as baseline for future
    comparisons.
"""
import sys
import os
import argparse
import json
import tqdm
from io import BytesIO
from base64 import b64decode
from PIL import Image, ImageDraw
from coords import bbox


def process_one(json_file: str, output_path: str):
    """Process a single JSON file to create a baseline version."""
    filename = os.path.basename(json_file)
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if data is None:
        print(f"Warning: JSON file '{json_file}' is empty or invalid.")
        return

    page = data.get('pages', [])[0] if data.get('pages') else None
    if page is None:
        print(f"Warning: No pages found in JSON file '{json_file}'.")
        return

    page_bb = bbox.default_bbox(page.get("bbox", {}))
    deskew = page.get("deskew", 0.0)
    rot = page.get("rotation", 0)

    if rot != 0:
        print(f"Warning: Rotation is not zero in JSON file '{json_file}'. Skipping.")
        return

    blocks = page.get("blocks", [])
    if 0 == len(blocks):
        print(f"Warning: No blocks found in JSON file '{json_file}'.")
        return
    lines = blocks[0].get('lines', [])

    bbs = []
    for line in lines:
        if 'words' in line:
            for word in line['words']:
                bb = bbox.default_bbox(word.get("bbox", {}))
                text = word.get("text", "")
                conf = word.get("confidence", 0.)
                if conf > 80. and len(text) > 3:
                    bbs.append(bb)

    if 0 == len(bbs):
        return

    store = {"bboxes": bbs, "deskew": deskew, "rotation": rot, "page_bbox": page_bb}
    output_file = os.path.join(output_path, f"{filename}.bboxes.json")
    with open(output_file, 'w', encoding='utf-8') as fout:
        json.dump(store, fout, cls=bbox.json_encoder)

    binary_img = page.get("image", {}).get("binary", None)
    if binary_img is None:
        print(f"Warning: No binary image found in JSON file '{json_file}'.")
        return

    img_base64 = binary_img["data"]
    page_image = Image.open(BytesIO(b64decode(img_base64)))
    page_image = page_image.convert("RGB")
    draw = ImageDraw.Draw(page_image)
    max_x, max_y = page_image.size
    for bb in bbs:
        xl = max(0, bb.xl)
        yt = max(0, bb.yt)
        xr = min(max_x, bb.xr)
        yb = min(max_y, bb.yb)
        draw.rectangle([xl, yt, xr, yb], outline="red", width=3)
    img_output_file = os.path.join(output_path, f"{filename}.bboxes.png")
    page_image.save(img_output_file)



def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(description='Read directory with QA results and create baseline for json files.')
    parser.add_argument('--input', help='Input QA DIR',
                        type=str, required=True)
    parser.add_argument('--output', help='output dir',
                        type=str, required=True, default=None)
    flags = parser.parse_args()

    dir_qa_path = flags.input
    output_path = flags.output

    # Check if input file exists
    if not os.path.exists(dir_qa_path):
        print(f"Error: Input file '{dir_qa_path}' does not exist!")
        sys.exit(1)

    os.makedirs(output_path, exist_ok=True)


    try:
        jsons_files = [os.path.join(dir_qa_path, filename) for filename in os.listdir(dir_qa_path) if filename.endswith('raw.json')]
        for json_file in tqdm.tqdm(jsons_files, desc="Processing JSON files"):
            process_one(json_file, output_path)

    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
