#!/usr/bin/env python3
"""
Compare two OCR results from different processing pipelines.
Result 1: qa_output - contains bounding boxes in JSON files
Result 2: qa_dp_advent - contains bounding boxes nested in page structure

The script:
1. Finds corresponding results from both directories
2. Normalizes bounding boxes to the same coordinate system
3. Compares bounding boxes and reports which result has more non-overlapping boxes
"""

import json
import os
import copy
import base64
import io
from typing import List, Dict, Tuple, Set
import re
from PIL import Image, ImageDraw
from coords.bbox import overlap_min, deskew_bboxes
from coords.bbox_wh import bbox

# Directories
RESULT1_DIR = r'D:\projects\issues\jira-515\newst_paddle\qa_output'
RESULT2_DIR = r'D:\projects\issues\jira-515\newst_paddle\qa_dp_advent'

def normalize_bbox(bb, deskew, original_size: Tuple[int, int],
                   target_size: Tuple[int, int]) -> Dict:
    """
    Normalize bounding box from original_size to target_size coordinate system.
    """
    orig_w, orig_h = original_size
    target_w, target_h = target_size
    
    bb = deskew_bboxes(deskew, bbox({'x':0, 'y':0, 'w': orig_w, 'h': orig_h}), [bb])[0]
    bb.scale(target_w / orig_w if orig_w > 0 else 1.)
    return bb


def load_result1_boxes(filepath: str) -> List[Dict]:
    """
    Load bounding boxes from result 1 (qa_output).
    Format: [bbox, ...]
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        boxes = json.load(f)
    bboxes = [bbox(box) for box in boxes]
    return bboxes


def load_result2_boxes(filepath: str) -> Tuple[List[Dict], Tuple[int, int], Image.Image, float, int]:
    """
    Load bounding boxes from result 2 (qa_dp_advent).
    Returns: (boxes, image_size)
    Format: boxes = [bboxes, ...]
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Get page information
    pages = data.get('pages', [])
    if not pages:
        return [], (0, 0), None, 0., 0
    
    page = pages[0]
    
    # Get image size from page bbox
    page_bbox = page.get('bbox', {})
    image_size = (page_bbox.get('w', 0), page_bbox.get('h', 0))

    images = page.get('image', {})
    bin_img = images.get("binary", {}).get("data", None)
    if bin_img is not None:
        png_buf = base64.b64decode(bin_img)
        data = io.BytesIO()
        data.write(png_buf)
        data.seek(0)
        bin_img = Image.open(data)
        
    deskew = page.get('deskew', 0.0)
    
    rotation = page.get('rotation', 0)
    
    # Extract all word bounding boxes from nested structure
    boxes = []
    for block in page.get('blocks', []):
        for line in block.get('lines', []):
            for word in line.get('words', []):
                bb = word.get('bbox', {})
                if bb:
                    boxes.append(bbox(bb))
    
    return boxes, image_size, bin_img, deskew, rotation


def get_image_size_from_result1(filepath: str) -> Tuple[int, int]:
    """
    Try to get image size from result 1.
    Since result 1 doesn't store image size, we'll infer it from the bounding boxes.
    """
    img_path = filepath.replace('.bbs.json', '')
    max_x, max_y = 0, 0
    #open with PIL to get image size
    with Image.open(img_path) as img:
        max_x, max_y = img.size
    # Add some padding since boxes might not reach the edges
    return (max_x, max_y)


def find_matching_files() -> List[Tuple[str, str]]:
    """
    Find matching files between result1 and result2 directories.
    Returns list of (result1_path, result2_path) tuples.
    """
    matches = []
    
    # Get all JSON files from result1
    result1_files = {}
    for filename in os.listdir(RESULT1_DIR):
        if filename.endswith('.bbs.json'):
            # Extract prefix: remove '_detected_lambda_*.png.bbs.json'
            match = re.match(r'^(.+?)_detected_lambda_.*\.png\.bbs\.json$', filename)
            if match:
                prefix = match.group(1)
                result1_files[prefix] = filename
    
    # Get all JSON files from result2
    result2_files = {}
    for filename in os.listdir(RESULT2_DIR):
        if filename.endswith('.raw.json'):
            # Extract prefix: remove '.raw.json'
            prefix = filename.replace('.raw.json', '')
            result2_files[prefix] = filename
    
    # Find matches
    for prefix in result1_files:
        if prefix in result2_files:
            result1_path = os.path.join(RESULT1_DIR, result1_files[prefix])
            result2_path = os.path.join(RESULT2_DIR, result2_files[prefix])
            matches.append((result1_path, result2_path, prefix))
    
    return matches


def compare_boxes(boxes1: List[Dict], boxes2: List[Dict], 
                  overlap_threshold: float = 0.5) -> Tuple[int, int, int, list, list]:
    """
    Compare two sets of bounding boxes.
    Returns: (boxes1_unique, boxes2_unique, common_boxes)
    
    A box is considered "matched" if it has IoU >= overlap_threshold with any box from the other set.
    """
    matched1 = set()
    matched2 = set()
    
    keep_bbs1 = copy.deepcopy(boxes1)
    keep_bbs2 = copy.deepcopy(boxes2)
    
    # For each box in boxes1, find if it matches any box in boxes2
    for i, box1 in enumerate(boxes1):
        for j, box2 in enumerate(boxes2):
            iou = overlap_min(box1, box2)
            if iou >= overlap_threshold:
                keep_bbs1[i] = None
                keep_bbs2[j] = None
                matched1.add(i)
                matched2.add(j)
    
    boxes1_unique = len(boxes1) - len(matched1)
    boxes2_unique = len(boxes2) - len(matched2)
    common_boxes = len(matched1)  # or len(matched2), should be similar
    
    return boxes1_unique, boxes2_unique, common_boxes, keep_bbs1, keep_bbs2


def main():
    """Main comparison function."""
    print("=" * 80)
    print("OCR Results Comparison")
    print("=" * 80)
    print(f"Result 1 directory: {RESULT1_DIR}")
    print(f"Result 2 directory: {RESULT2_DIR}")
    print()
    
    # Find matching files
    matches = find_matching_files()
    print(f"Found {len(matches)} matching file pairs")
    print()
    
    if not matches:
        print("No matching files found!")
        return
    
    # Compare each pair
    results = []
    
    for result1_path, result2_path, prefix in matches:
        print(f"Processing: {prefix}")
        print(f"  Result 1: {os.path.basename(result1_path)}")
        print(f"  Result 2: {os.path.basename(result2_path)}")
        
        try:
            # Load boxes from result 1
            boxes1 = load_result1_boxes(result1_path)
            
            # Load boxes from result 2
            boxes2, image_size2, pil_img2, deskew, rotation = load_result2_boxes(result2_path)
            
            if rotation != 0:
                print(f"  WARNING: Result 2 has rotation {rotation} degrees, which is not handled.")
                continue
            
            if not boxes1 or not boxes2:
                print(f"  WARNING: Empty results (boxes1={len(boxes1)}, boxes2={len(boxes2)})")
                print()
                continue
            
            # Get image size from result 1 (inferred from boxes)
            image_size1 = get_image_size_from_result1(result1_path)
            if image_size1 == (0, 0):
                print("  WARNING: Could not determine image size for Result 1")
                continue
            
            print(f"  Result 1: {len(boxes1)} boxes, inferred size: {image_size1}")
            print(f"  Result 2: {len(boxes2)} boxes, actual size: {image_size2}")
            
            # Normalize boxes to a common coordinate system (use result2's size as reference)
            normalized_boxes1 = []
            for box in boxes1:
                normalized_box = normalize_bbox(box, deskew, image_size1, image_size2)
                normalized_boxes1.append(normalized_box)
            
            # boxes2 are already in the correct coordinate system
            normalized_boxes2 = boxes2
            
            # Compare boxes
            unique1, unique2, common, kept_bb1, kept_bb2 = compare_boxes(
                normalized_boxes1, normalized_boxes2, overlap_threshold=0.5
            )
            
            # draw comparision results
            if pil_img2 is not None and (unique1 > 10 or unique2 > 10):
                store = False
                pil_img2 = pil_img2.convert("RGB")
                draw = ImageDraw.Draw(pil_img2)
                for b in kept_bb1:
                    if b is not None:
                        draw.rectangle([b.xl, b.yt, b.xr, b.yb], outline='red', width=3)
                        store = True
                for b in kept_bb2:
                    if b is not None:
                        draw.rectangle([b.xl, b.yt, b.xr, b.yb], outline='blue', width=3)
                        store = True
                if store:
                    debug_img_path = os.path.join(os.path.dirname(__file__), "__cmp")
                    debug_img_path = os.path.join(debug_img_path, f"{prefix}_comparison.png")
                    pil_img2.save(debug_img_path)
                    print(f"  Debug comparison image saved to: {debug_img_path}")
            
            
            print(f"  Comparison results:")
            print(f"    Common boxes (50%+ overlap): {common}")
            print(f"    Unique to Result 1: {unique1}")
            print(f"    Unique to Result 2: {unique2}")
            
            # Determine which result has more unique boxes
            if unique1 > unique2:
                winner = "Result 1 (qa_output)"
                diff = unique1 - unique2
            elif unique2 > unique1:
                winner = "Result 2 (qa_dp_advent)"
                diff = unique2 - unique1
            else:
                winner = "Tie"
                diff = 0
            
            print(f"    More unique boxes in: {winner} (difference: {diff})")
            print()
            
            results.append({
                'prefix': prefix,
                'boxes1': len(boxes1),
                'boxes2': len(boxes2),
                'common': common,
                'unique1': unique1,
                'unique2': unique2,
                'winner': winner,
                'diff': diff
            })
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            print()
            continue
    
    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    
    if not results:
        print("No results to summarize.")
        return
    
    # Save detailed results to CSV
    csv_path = os.path.join(os.path.dirname(__file__), 'comparison_results.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("prefix,total_boxes_result1,total_boxes_result2,common_boxes,unique_result1,unique_result2,winner,difference\n")
        for r in results:
            f.write(f"{r['prefix']},{r['boxes1']},{r['boxes2']},{r['common']},{r['unique1']},{r['unique2']},{r['winner']},{r['diff']}\n")
    
    print(f"Detailed results saved to: {csv_path}")
    print()
    
    # Statistics
    result1_wins = sum(1 for r in results if 'Result 1' in r['winner'])
    result2_wins = sum(1 for r in results if 'Result 2' in r['winner'])
    ties = sum(1 for r in results if r['winner'] == 'Tie')
    
    print(f"Total comparisons: {len(results)}")
    print(f"Result 1 (qa_output) has more unique boxes: {result1_wins} times")
    print(f"Result 2 (qa_dp_advent) has more unique boxes: {result2_wins} times")
    print(f"Ties: {ties} times")
    print()
    
    # Show cases with biggest differences
    print("Top 10 cases with most unique boxes in Result 1:")
    sorted_by_r1 = sorted(results, key=lambda x: x['unique1'], reverse=True)[:10]
    for i, r in enumerate(sorted_by_r1, 1):
        print(f"  {i}. {r['prefix']}: {r['unique1']} unique boxes")
    print()
    
    print("Top 10 cases with most unique boxes in Result 2:")
    sorted_by_r2 = sorted(results, key=lambda x: x['unique2'], reverse=True)[:10]
    for i, r in enumerate(sorted_by_r2, 1):
        print(f"  {i}. {r['prefix']}: {r['unique2']} unique boxes")
    print()
    
    print("=" * 80)


if __name__ == '__main__':
    main()
