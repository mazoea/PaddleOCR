#!/usr/bin/env python3
"""
Script to compare QA baseline results with text spotting output results.

This script:
1. Reads baseline bounding boxes from qa_dp_advent_baseline directory
   - Files contain: bboxes (list of {x, y, w, h}), page_bbox, deskew angle, rotation
2. Reads text spotting results from qa_output directory
   - Files contain: bboxes (list of {x, y, w, h}), img_w, img_h
3. Transforms the text spotting results to baseline coordinate system:
   - Scales from output image dimensions to baseline page dimensions
   - Applies deskew transformation using the coords library
4. Compares bounding boxes to find how many baseline boxes are not covered
   - Uses overlap threshold (default 50%) to determine if a bbox is covered
5. Outputs comprehensive statistics about the coverage

Usage:
    python cmp_ts_rs_with_qa_baseline.py [options]
    
    Options:
        --baseline-dir PATH       Path to baseline directory (default: qa_dp_advent_baseline)
        --output-dir PATH         Path to output directory (default: qa_output)
        --overlap-threshold NUM   Overlap ratio 0.0-1.0 to consider covered (default: 0.5)
        --additional              Show detailed per-file statistics
    
    Examples:
        # Run with defaults
        python cmp_ts_rs_with_qa_baseline.py
        
        # Custom overlap threshold
        python cmp_ts_rs_with_qa_baseline.py --overlap-threshold 0.7
        
        # Show additional statistics
        python cmp_ts_rs_with_qa_baseline.py --additional
        
        # Custom directories
        python cmp_ts_rs_with_qa_baseline.py --baseline-dir path/to/baseline --output-dir path/to/output

Output:
    - Overall summary with total uncovered bboxes (main metric)
    - Per-file statistics showing coverage for each document (with --additional)
    - Additional statistics including low-coverage files and averages (with --additional)
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import tqdm
import argparse
from PIL import Image, ImageDraw

# Import coords library for transformations
from coords import bbox, angle


def load_baseline_json(filepath: str) -> Dict:
    """Load baseline JSON with bboxes, deskew, rotation, and page_bbox."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_output_json(filepath: str) -> Dict:
    """Load text spotting output JSON with bboxes and image dimensions."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def transform_output_to_baseline(output_bboxes: List[Dict], 
                                  output_img_w: int, 
                                  output_img_h: int,
                                  baseline_page_bbox: Dict,
                                  baseline_deskew: float,
                                  baseline_rotation: float) -> List[bbox.default_bbox]:
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
    # Calculate scaling factors
    baseline_w = baseline_page_bbox['w']
    baseline_h = baseline_page_bbox['h']
    
    scale_x = baseline_w / output_img_w
    scale_y = baseline_h / output_img_h
    
    transformed_bboxes = []
    
    for bb in output_bboxes:
        # Create bbox object and scale to baseline dimensions
        scaled_bbox = bbox.default_bbox({
            'x': bb['x'] * scale_x,
            'y': bb['y'] * scale_y,
            'w': bb['w'] * scale_x,
            'h': bb['h'] * scale_y
        })
        
        # Apply deskew if needed
        if baseline_deskew != 0.0 or baseline_rotation != 0:
            # Create page bbox for deskewing
            page_bb = bbox.default_bbox(baseline_page_bbox)

            bbox.default_bbox(page_bb)
            deskewed_bbox = bbox.deskew_bboxes(baseline_deskew, page_bb, [scaled_bbox])[0]
            transformed_bboxes.append(deskewed_bbox)
        else:
            transformed_bboxes.append(scaled_bbox)
    
    return transformed_bboxes


def is_bbox_covered(baseline_bbox: Dict, 
                    output_bboxes: List[bbox.default_bbox],
                    overlap_threshold: float = 0.5) -> bool:
    """
    Check if a baseline bbox is covered by any output bbox.
    
    A baseline bbox is considered covered if it has sufficient overlap with 
    at least one output bbox.
    
    Args:
        baseline_bbox: Baseline bbox in format {x, y, w, h}
        output_bboxes: List of transformed output bboxes
        overlap_threshold: Minimum overlap ratio to consider covered (0.0-1.0)
        
    Returns:
        True if covered, False otherwise
    """
    baseline_bb = bbox.default_bbox(baseline_bbox)
    
    for output_bb in output_bboxes:
        # Calculate overlap ratio (overlap area / smaller bbox area)
        overlap_ratio = bbox.overlap_min(baseline_bb, output_bb)
        
        if overlap_ratio >= overlap_threshold:
            return True
    
    return False


def get_file_prefix(filename: str) -> str:
    """
    Extract file prefix for matching baseline and output files.
    
    Examples:
    - '37.pdf-000001.png.raw.json.bboxes.json' -> '37.pdf-000001.png'
    - '37.pdf-000001.png_detection.png.bbs.json' -> '37.pdf-000001.png'
    """
    # Remove various suffixes
    name = filename
    
    # For baseline files: remove .raw.json.bboxes.json
    if '.raw.json.bboxes.json' in name:
        return name.replace('.raw.json.bboxes.json', '')
    
    # For output files: remove _detection.png.bbs.json
    if '_detection.png.bbs.json' in name:
        return name.replace('_detection.png.bbs.json', '')
    
    # Fallback: remove extension
    return os.path.splitext(name)[0]


def find_matching_files(baseline_dir: str, output_dir: str) -> List[Tuple[str, str]]:
    """
    Find matching pairs of baseline and output files based on file prefix.
    
    Returns:
        List of tuples (baseline_path, output_path)
    """
    baseline_files = {}
    for f in os.listdir(baseline_dir):
        if f.endswith('.json') and not f.endswith('.png'):
            prefix = get_file_prefix(f)
            baseline_files[prefix] = os.path.join(baseline_dir, f)
    
    output_files = {}
    for f in os.listdir(output_dir):
        if f.endswith('.bbs.json'):
            prefix = get_file_prefix(f)
            output_files[prefix] = os.path.join(output_dir, f)
    
    # Find matching pairs
    matching_pairs = []
    for prefix in baseline_files:
        if prefix in output_files:
            matching_pairs.append((baseline_files[prefix], output_files[prefix]))
        else:
            print(f"Warning: No output file found for baseline: {prefix}")
    
    # Report output files without baseline
    for prefix in output_files:
        if prefix not in baseline_files:
            print(f"Warning: No baseline file found for output: {prefix}")
    
    return matching_pairs


def compare_files(baseline_path: str, 
                  output_path: str, 
                  overlap_threshold: float = 0.5) -> Dict:
    """
    Compare a single pair of baseline and output files.
    
    Returns:
        Dictionary with comparison statistics
    """
    # Load data
    baseline_data = load_baseline_json(baseline_path)
    output_data = load_output_json(output_path)
    
    baseline_bboxes = baseline_data.get('bboxes', [])
    output_bboxes = output_data.get('bboxes', [])
    
    # Get dimensions and transformations
    page_bbox = baseline_data.get('page_bbox', {'x': 0, 'y': 0, 'w': 1, 'h': 1})
    deskew = baseline_data.get('deskew', 0.0)
    rotation = baseline_data.get('rotation', 0)

    if 0 != rotation:
        raise NotImplementedError("Rotation handling is not implemented in this script.")
    
    output_img_w = output_data.get('img_w', 1)
    output_img_h = output_data.get('img_h', 1)
    
    # Transform output bboxes to baseline coordinate system
    transformed_output_bboxes = transform_output_to_baseline(
        output_bboxes,
        output_img_w,
        output_img_h,
        page_bbox,
        deskew,
        rotation
    )
    
    # Check coverage
    uncovered_count = 0
    covered_count = 0

    to_show_bbs = []
    
    for baseline_bb in baseline_bboxes:
        if is_bbox_covered(baseline_bb, transformed_output_bboxes, overlap_threshold):
            covered_count += 1
        else:
            uncovered_count += 1
            to_show_bbs.append(baseline_bb)

    #if 0 != len(to_show_bbs):
    #    png_file = baseline_path[:-len('.json')]+ '.png'
    #    img = Image.open(png_file).convert('RGB')
    #    draw = ImageDraw.Draw(img)
    #    for bb in to_show_bbs:
    #        bb = bbox.default_bbox(bb)
    #        draw.rectangle([bb.xl, bb.yt, bb.xr, bb.yb], outline='blue', width=4)
    #    img.save(f"{png_file}.uncovered.png")
    
    # Calculate statistics
    total_baseline = len(baseline_bboxes)
    total_output = len(output_bboxes)
    coverage_percentage = (covered_count / total_baseline * 100) if total_baseline > 0 else 0
    
    return {
        'baseline_path': baseline_path,
        'output_path': output_path,
        'total_baseline_bboxes': total_baseline,
        'total_output_bboxes': total_output,
        'covered_bboxes': covered_count,
        'uncovered_bboxes': uncovered_count,
        'coverage_percentage': coverage_percentage,
        'scale_x': page_bbox['w'] / output_img_w,
        'scale_y': page_bbox['h'] / output_img_h,
        'deskew': deskew,
        'rotation': rotation
    }


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare QA baseline results with text spotting output results.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--baseline-dir',
        type=str,
        default=r"d:\projects\issues\jira-515\newst_paddle\qa_dp_advent_baseline",
        help='Directory containing baseline JSON files with bboxes'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=r"d:\projects\issues\jira-515\newst_paddle\qa_output",
        help='Directory containing text spotting output JSON files'
    )
    parser.add_argument(
        '--overlap-threshold',
        type=float,
        default=0.5,
        help='Minimum overlap ratio (0.0-1.0) to consider a bbox as covered'
    )
    parser.add_argument(
        '--additional',
        action='store_true',
        default=False,
        help='Show additional detailed statistics (per-file breakdown and extra info)'
    )
    
    return parser.parse_args()


def main():
    """Main function to run the comparison."""
    # Parse command line arguments
    args = parse_arguments()
    
    baseline_dir = args.baseline_dir
    output_dir = args.output_dir
    overlap_threshold = args.overlap_threshold
    additional = args.additional
    
    print(f"Comparing results...")
    print(f"Baseline directory: {baseline_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Overlap threshold: {overlap_threshold}")
    print(f"Additional statistics: {additional}")
    print("-" * 80)
    
    # Find matching file pairs
    matching_pairs = find_matching_files(baseline_dir, output_dir)
    print(f"\nFound {len(matching_pairs)} matching file pairs")
    print("-" * 80)
    
    # Compare each pair
    results = []
    for baseline_path, output_path in tqdm.tqdm(matching_pairs):
        try:
            result = compare_files(baseline_path, output_path, overlap_threshold)
            results.append(result)
        except Exception as e:
            print(f"Error processing {os.path.basename(baseline_path)}: {e}")
            import traceback
            traceback.print_exc()
    
    # Calculate overall statistics
    total_baseline_bboxes = sum(r['total_baseline_bboxes'] for r in results)
    total_output_bboxes = sum(r['total_output_bboxes'] for r in results)
    total_covered = sum(r['covered_bboxes'] for r in results)
    total_uncovered = sum(r['uncovered_bboxes'] for r in results)
    
    overall_coverage = (total_covered / total_baseline_bboxes * 100) if total_baseline_bboxes > 0 else 0
    
    # Print summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"Total files compared: {len(results)}")
    print(f"Total baseline bboxes: {total_baseline_bboxes}")
    print(f"Total output bboxes: {total_output_bboxes}")
    print(f"Total covered bboxes: {total_covered}")
    print(f"Total uncovered bboxes: {total_uncovered}")
    print(f"Overall coverage: {overall_coverage:.2f}%")
    print("-" * 80)

    if additional:

        # Print per-file statistics
        print("\nPER-FILE STATISTICS")
        print("-" * 80)
        print(f"{'File':<40} {'Baseline':<10} {'Output':<10} {'Covered':<10} {'Uncovered':<10} {'Coverage %':<12}")
        print("-" * 80)

        # Sort by coverage percentage (lowest first to highlight problems)
        sorted_results = sorted(results, key=lambda x: x['coverage_percentage'])

        for r in sorted_results:
            filename = os.path.basename(r['baseline_path']).replace('.raw.json.bboxes.json', '')
            print(f"{filename:<40} {r['total_baseline_bboxes']:<10} {r['total_output_bboxes']:<10} "
                  f"{r['covered_bboxes']:<10} {r['uncovered_bboxes']:<10} {r['coverage_percentage']:<12.2f}")

        # Additional statistics
        print("\n" + "=" * 80)
        print("ADDITIONAL STATISTICS")
        print("=" * 80)

        # Files with low coverage
        low_coverage_files = [r for r in results if r['coverage_percentage'] < 80]
        if low_coverage_files:
            print(f"\nFiles with coverage < 80%: {len(low_coverage_files)}")
            for r in low_coverage_files:
                filename = os.path.basename(r['baseline_path']).replace('.raw.json.bboxes.json', '')
                print(f"  - {filename}: {r['coverage_percentage']:.2f}%")

        # Files with perfect coverage
        perfect_coverage_files = [r for r in results if r['coverage_percentage'] == 100]
        if perfect_coverage_files:
            print(f"\nFiles with 100% coverage: {len(perfect_coverage_files)}")

        # Average bboxes per file
        avg_baseline = total_baseline_bboxes / len(results) if results else 0
        avg_output = total_output_bboxes / len(results) if results else 0
        print(f"\nAverage baseline bboxes per file: {avg_baseline:.2f}")
        print(f"Average output bboxes per file: {avg_output:.2f}")

        # Bbox count difference
        bbox_diff = total_output_bboxes - total_baseline_bboxes
        bbox_diff_pct = (bbox_diff / total_baseline_bboxes * 100) if total_baseline_bboxes > 0 else 0
        print(f"\nTotal bbox count difference: {bbox_diff:+d} ({bbox_diff_pct:+.2f}%)")

        print("\n" + "=" * 80)
        print(f"Main metric: {total_uncovered} baseline bboxes are NOT covered by output bboxes")
        print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
