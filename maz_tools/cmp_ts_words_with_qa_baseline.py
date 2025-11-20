#!/usr/bin/env python3
"""
Script to compare QA baseline results with text spotting ts_words from raw.json files.

This script:
1. Reads baseline bounding boxes from qa_dp_advent_baseline directory
   - Files contain: bboxes (list of {x, y, w, h}), page_bbox, deskew angle, rotation
2. Reads text spotting ts_words from raw.json files
   - Files contain: ts_words key with list of bounding boxes
3. Transforms the ts_words bboxes to baseline coordinate system:
   - Scales from output image dimensions to baseline page dimensions
   - Applies deskew transformation using the coords library
4. Compares bounding boxes to find how many baseline boxes are not covered
   - Uses overlap threshold (default 50%) to determine if a bbox is covered
5. Outputs comprehensive statistics about the coverage

Usage:
    python cmp_ts_words_with_qa_baseline.py [options]
    
    Options:
        --raw-json-dir PATH       Path to directory with raw.json files (default: d:\projects\issues\jira-515\newst_paddle)
        --baseline-dir PATH       Path to baseline directory (default: d:\projects\issues\jira-515\newst_paddle\qa_dp_advent_baseline)
        --overlap-threshold NUM   Overlap ratio 0.0-1.0 to consider covered (default: 0.5)
        --show-details            Show detailed per-file statistics
    
    Examples:
        # Run with defaults
        python cmp_ts_words_with_qa_baseline.py
        
        # Custom overlap threshold
        python cmp_ts_words_with_qa_baseline.py --overlap-threshold 0.7
        
        # Show detailed statistics
        python cmp_ts_words_with_qa_baseline.py --show-details
        
        # Custom directories
        python cmp_ts_words_with_qa_baseline.py --raw-json-dir path/to/raw_jsons --baseline-dir path/to/baseline

Output:
    - Overall summary with total uncovered bboxes (main metric)
    - Coverage rate percentage
    - Per-file statistics showing coverage for each document (with --show-details)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import sys
from tqdm import tqdm
import argparse

# Import coords library for transformations
try:
    from coords import bbox, angle
except ImportError:
    print("Error: coords library not found. Please install it first.")
    sys.exit(1)


def load_baseline_json(filepath: str) -> Dict:
    """Load baseline JSON with bboxes, deskew, rotation, and page_bbox."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_raw_json(filepath: str) -> Dict:
    """Load raw.json file with ts_words key containing bounding boxes."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def extract_ts_words_bboxes(raw_json_data: Dict) -> Tuple[List[Dict], int, int]:
    """
    Extract bounding boxes from ts_words key and get image dimensions.
    
    Args:
        raw_json_data: Raw JSON data with ts_words key
        
    Returns:
        Tuple of (bboxes_list, img_width, img_height)
    """
    bboxes = []

    pages = raw_json_data.get('pages', [])
    if 0 == len(pages):
        return bboxes, 0, 0
    page = pages[0]
    ia = page.get('ia', {})
    # Get ts_words
    ts_words = ia.get('ts_words', [])
    if 0 == len(ts_words):
        return bboxes, 0, 0
    
    # Extract bboxes from ts_words
    for bb in ts_words:
        bboxes.append(bbox.default_bbox(bb))

    # Get image dimensions
    img_w = raw_json_data.get('img_w', page.get('bbox', {}).get('w', 0))
    img_h = raw_json_data.get('img_h', page.get('bbox', {}).get('h', 0))
    
    return bboxes, img_w, img_h


def transform_output_to_baseline(output_bboxes: List[bbox.default_bbox],
                                  output_img_w: int, 
                                  output_img_h: int,
                                  baseline_page_bbox: Dict,
                                  baseline_deskew: float,
                                  baseline_rotation: float) -> List:
    """
    Transform output bounding boxes to baseline coordinate system.
    
    Steps:
    1. Scale from output image dimensions to baseline dimensions
    2. Apply deskew transformation
    
    Args:
        output_bboxes: List of bboxes from output (format: {x, y, w, h})
        output_img_w: Width of output image
        output_img_h: Height of output image
        baseline_page_bbox: Page bbox from baseline (format: {x, y, w, h})
        baseline_deskew: Deskew angle in degrees from baseline
        baseline_rotation: Rotation angle in degrees from baseline
        
    Returns:
        List of transformed bboxes in baseline coordinate system
    """
    if output_img_w == 0 or output_img_h == 0:
        print("Warning: Invalid image dimensions (0), cannot transform bboxes")
        return []
    
    # Calculate scaling factors
    baseline_w = baseline_page_bbox['w']
    baseline_h = baseline_page_bbox['h']
    
    scale_x = baseline_w / output_img_w

    # Convert output bboxes to coords library format and scale
    scaled_bboxes = []
    for bb in output_bboxes:
        bb.scale(scale_x)  # Scale width and height
        if baseline_deskew != 0:
            bb = bbox.deskew_bboxes(baseline_deskew, bbox.default_bbox({'x':0, 'y':0, 'w': output_img_w, 'h': output_img_h}), [bb])[0]
        scaled_bboxes.append(bb)

    return scaled_bboxes


def is_bbox_covered(baseline_bb, output_bboxes: List, threshold: float = 0.5) -> bool:
    """
    Check if a baseline bbox is covered by any output bbox.
    
    Uses overlap_min to calculate the coverage ratio.
    
    Args:
        baseline_bb: Baseline bounding box
        output_bboxes: List of output bounding boxes
        threshold: Minimum overlap ratio to consider as covered
        
    Returns:
        bool: True if covered, False otherwise
    """
    for output_bb in output_bboxes:
        try:
            overlap = bbox.overlap_min(baseline_bb, output_bb)
            if overlap >= threshold:
                return True
        except Exception:
            continue
    
    return False


def find_matching_files(baseline_dir: str, raw_json_dir: str) -> List[Tuple[str, str]]:
    """
    Find matching files between baseline and raw.json directories.
    
    Args:
        baseline_dir: Path to baseline directory
        raw_json_dir: Path to raw.json directory
        
    Returns:
        List of tuples (baseline_path, raw_json_path)
    """
    matches = []
    
    # Get all baseline JSON files
    baseline_files = {}
    for filename in os.listdir(baseline_dir):
        if filename.endswith('.json'):
            # Extract base name without extension
            base_name = filename[:-5]  # Remove .json
            # Also try removing common suffixes
            for suffix in ['.raw.json.bboxes', '.bboxes']:
                if base_name.endswith(suffix):
                    base_name = base_name[:-len(suffix)]
            baseline_files[base_name] = os.path.join(baseline_dir, filename)
    
    # Get all raw.json files
    raw_json_files = {}
    for filename in os.listdir(raw_json_dir):
        if filename.endswith('.raw.json'):
            # Extract base name
            base_name = filename[:-9]  # Remove .raw.json
            raw_json_files[base_name] = os.path.join(raw_json_dir, filename)
    
    # Find matches
    for base_name in baseline_files:
        if base_name in raw_json_files:
            matches.append((baseline_files[base_name], raw_json_files[base_name]))
    
    return matches


def compare_files(baseline_path: str, raw_json_path: str, overlap_threshold: float) -> Dict:
    """
    Compare baseline and raw.json file, reporting uncovered bboxes.
    
    Args:
        baseline_path: Path to baseline JSON file
        raw_json_path: Path to raw.json file
        overlap_threshold: Minimum overlap to consider as covered
        
    Returns:
        dict: Statistics about the comparison
    """
    # Load baseline data
    baseline_data = load_baseline_json(baseline_path)
    baseline_bboxes = baseline_data.get('bboxes', [])

    # Load raw.json data
    raw_json_data = load_raw_json(raw_json_path)
    ts_words_bboxes, img_w, img_h = extract_ts_words_bboxes(raw_json_data)

    if 0 == len(ts_words_bboxes):
        return None
    
    # Compare
    covered_count = 0
    uncovered_count = 0

    uncovered_bbs = []
    
    for baseline_bb in baseline_bboxes:
        baseline_bb = bbox.default_bbox(baseline_bb)
        if is_bbox_covered(baseline_bb, ts_words_bboxes, overlap_threshold):
            covered_count += 1
        else:
            uncovered_count += 1
            uncovered_bbs.append(baseline_bb)

    if uncovered_count > 0:
        dump_path = Path("d:/projects/issues/jira-515/tmp")
        # show uncovered bboxes
        png_file = baseline_path[:-len('.json')]+ '.png'
        from PIL import Image, ImageDraw
        img = Image.open(png_file).convert('RGB')
        draw = ImageDraw.Draw(img)
        for bb in uncovered_bbs:
            draw.rectangle([bb.xl, bb.yt, bb.xr, bb.yb], outline='blue', width=4)
        file_name = os.path.basename(png_file)
        dump_png_path = os.path.join(dump_path,f"{file_name}.uncovered.png")
        img.save(f"{dump_png_path}.uncovered.png")
    
    return {
        'filename': os.path.basename(raw_json_path),
        'total_baseline_bboxes': len(baseline_bboxes),
        'ts_words_bboxes': len(ts_words_bboxes),
        'covered_bboxes': covered_count,
        'uncovered_bboxes': uncovered_count,
        'coverage_rate': 100.0 * covered_count / len(baseline_bboxes) if len(baseline_bboxes) > 0 else 0.0
    }


def main():
    parser = argparse.ArgumentParser(
        description='Compare QA baseline with text spotting ts_words from raw.json files',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--raw-json-dir',
        type=str,
        default=r'd:\projects\issues\jira-515\newst_paddle',
        help='Path to directory containing raw.json files'
    )
    parser.add_argument(
        '--baseline-dir',
        type=str,
        default=r'd:\projects\issues\jira-515\newst_paddle\qa_dp_advent_baseline',
        help='Path to baseline directory'
    )
    parser.add_argument(
        '--overlap-threshold',
        type=float,
        default=0.5,
        help='Minimum overlap ratio to consider a bbox as covered (default: 0.5)'
    )
    parser.add_argument(
        '--show-details',
        action='store_true',
        help='Show detailed per-file statistics'
    )
    
    args = parser.parse_args()
    
    # Check if directories exist
    if not os.path.exists(args.baseline_dir):
        print(f"Error: Baseline directory not found: {args.baseline_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.raw_json_dir):
        print(f"Error: Raw JSON directory not found: {args.raw_json_dir}")
        sys.exit(1)
    
    print("Finding matching files...")
    matches = find_matching_files(args.baseline_dir, args.raw_json_dir)
    
    if len(matches) == 0:
        print("Error: No matching files found between baseline and raw.json directories")
        sys.exit(1)
    
    print(f"Found {len(matches)} matching file pairs")
    
    # Process all matches
    all_stats = []
    total_baseline_bboxes = 0
    total_covered_bboxes = 0
    total_uncovered_bboxes = 0
    
    print(f"\nComparing with overlap threshold: {args.overlap_threshold}")
    
    for baseline_path, raw_json_path in tqdm(matches, desc="Processing files"):
        try:
            stats = compare_files(baseline_path, raw_json_path, args.overlap_threshold)
            if stats is None:
                print(f"\nWarning: Skipping {os.path.basename(raw_json_path)}.")
                continue
            all_stats.append(stats)
            
            total_baseline_bboxes += stats['total_baseline_bboxes']
            total_covered_bboxes += stats['covered_bboxes']
            total_uncovered_bboxes += stats['uncovered_bboxes']
            
        except Exception as e:
            print(f"\nError processing {os.path.basename(raw_json_path)}: {e}")
            continue
    
    # Print results
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"Total files processed: {len(all_stats)}")
    print(f"Total baseline bounding boxes: {total_baseline_bboxes}")
    print(f"Covered bounding boxes: {total_covered_bboxes}")
    print(f"Uncovered bounding boxes: {total_uncovered_bboxes}")
    
    if total_baseline_bboxes > 0:
        overall_coverage = 100.0 * total_covered_bboxes / total_baseline_bboxes
        print(f"\nOverall coverage rate: {overall_coverage:.2f}%")
    
    if args.show_details and all_stats:
        print("\n" + "="*60)
        print("PER-FILE STATISTICS")
        print("="*60)
        
        # Sort by uncovered count (descending)
        sorted_stats = sorted(all_stats, key=lambda x: x['uncovered_bboxes'], reverse=True)
        
        print(f"\n{'Filename':<50} {'Baseline':<10} {'TS Words':<10} {'Covered':<10} {'Uncovered':<12} {'Coverage %':<12}")
        print("-" * 110)
        
        for stats in sorted_stats:
            print(f"{stats['filename']:<50} "
                  f"{stats['total_baseline_bboxes']:<10} "
                  f"{stats['ts_words_bboxes']:<10} "
                  f"{stats['covered_bboxes']:<10} "
                  f"{stats['uncovered_bboxes']:<12} "
                  f"{stats['coverage_rate']:<12.2f}")
        
        # Show files with low coverage (< 80%)
        low_coverage_files = [s for s in all_stats if s['coverage_rate'] < 80.0]
        if low_coverage_files:
            print(f"\n{len(low_coverage_files)} files with coverage < 80%:")
            for stats in sorted(low_coverage_files, key=lambda x: x['coverage_rate']):
                print(f"  {stats['filename']}: {stats['coverage_rate']:.2f}%")
    
    print("="*60)


if __name__ == '__main__':
    main()
