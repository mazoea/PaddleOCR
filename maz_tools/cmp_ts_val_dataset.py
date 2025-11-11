#!/usr/bin/env python3
"""
Compare validation dataset with validated model output.
Measures how many bounding boxes from the baseline dataset are not covered in the validated output.
"""

import os
import sys
import json
import argparse
from tqdm import tqdm

# Import coords library for bbox operations
try:
    from coords import bbox
except ImportError:
    print("Error: coords library not found. Please install it first.")
    sys.exit(1)


def parse_label_txt(label_txt_path):
    """
    Parse Label.txt file from PaddleOCR dataset format.
    
    Format: filename\t[{"transcription": text, "points": [[x,y], ...], "difficult": bool}, ...]
    
    Returns:
        dict: {filename: [bbox1, bbox2, ...]} where each bbox is in default_bbox format
    """
    data = {}
    
    with open(label_txt_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                # Split by tab to separate filename and annotations
                parts = line.split('\t', 1)
                if len(parts) != 2:
                    print(f"Warning: Line {line_num} has invalid format, skipping")
                    continue
                    
                filename = parts[0]
                annotations_json = parts[1]
                
                # Parse the JSON annotations
                annotations = json.loads(annotations_json)
                
                # Convert each annotation to bbox format
                bboxes = []
                for ann in annotations:
                    points = ann.get('points', [])
                    if len(points) != 4:
                        continue  # Skip invalid annotations
                    
                    # Convert 4-point polygon to {x, y, w, h} format
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    x = min(xs)
                    y = min(ys)
                    w = max(xs) - x
                    h = max(ys) - y
                    
                    # Create default_bbox
                    bb = bbox.default_bbox({"x":x, "y":y, "w":w, "h":h})
                    bboxes.append(bb)
                
                # Extract just the filename without directory
                filename_only = os.path.basename(filename)
                data[filename_only] = bboxes
                
            except json.JSONDecodeError as e:
                print(f"Warning: Line {line_num} has invalid JSON: {e}")
                continue
            except Exception as e:
                print(f"Warning: Error processing line {line_num}: {e}")
                continue
    
    return data


def load_output_json(output_dir):
    """
    Load validated output JSON files from directory.
    
    Expected format: filename.json with structure similar to the reference script.
    Returns:
        dict: {filename: [bbox1, bbox2, ...]}
    """
    data = {}
    
    if not os.path.exists(output_dir):
        print(f"Warning: Output directory does not exist: {output_dir}")
        return data
    
    for filename in os.listdir(output_dir):
        if not filename.endswith('.json'):
            continue
            
        filepath = os.path.join(output_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            # Extract bboxes from the JSON structure
            # Adjust this based on actual output format
            bboxes = []
            
            # Try different possible structures
            if isinstance(content, list):
                # List of bboxes
                for item in content:
                    if 'bbox' in item:
                        bb_data = item['bbox']
                    elif 'points' in item:
                        # Convert points to bbox
                        points = item['points']
                        xs = [p[0] for p in points]
                        ys = [p[1] for p in points]
                        bb_data = {
                            'x': min(xs),
                            'y': min(ys),
                            'w': max(xs) - min(xs),
                            'h': max(ys) - min(ys)
                        }
                    else:
                        continue
                    
                    bb = bbox.default_bbox(**bb_data)
                    bboxes.append(bb)
                    
            elif isinstance(content, dict):
                # Dictionary with bboxes key
                if 'bboxes' in content:
                    for bb_data in content['bboxes']:
                        bb = bbox.default_bbox(bb_data)
                        bboxes.append(bb)
                elif 'bbox' in content:
                    bb = bbox.default_bbox(**content['bbox'])
                    bboxes.append(bb)
            
            # Store with base filename (remove any suffixes like .json)
            base_filename = filename
            # Try to match with image extensions
            for ext in ['.json', '.bbs.json', '.detection.json']:
                if base_filename.endswith(ext):
                    base_filename = base_filename[:-len(ext)]
            
            data[base_filename] = bboxes
            
        except Exception as e:
            print(f"Warning: Error loading {filename}: {e}")
            continue
    
    return data


def is_bbox_covered(baseline_bb, output_bboxes, threshold=0.5):
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


def compare_files(baseline_data, output_data, overlap_threshold=0.5):
    """
    Compare baseline and output data, reporting uncovered bboxes.
    
    Args:
        baseline_data: Dictionary of {filename: [bboxes]}
        output_data: Dictionary of {filename: [bboxes]}
        overlap_threshold: Minimum overlap to consider as covered
        
    Returns:
        dict: Statistics about the comparison
    """
    stats = {
        'total_baseline_files': len(baseline_data),
        'total_output_files': len(output_data),
        'matched_files': 0,
        'unmatched_files': 0,
        'total_baseline_bboxes': 0,
        'total_covered_bboxes': 0,
        'total_uncovered_bboxes': 0,
        'uncovered_by_file': {}
    }
    
    # Find matching files
    baseline_files = set(baseline_data.keys())
    output_files = set(output_data.keys())

    def find_match(name, candidates):
        for cand in candidates:
            if name in cand:
                return name, cand
        return None
    
    matched = [find_match(name, output_files) for name in baseline_files]
    # remove None entries
    matched = [item for item in matched if item is not None]
    stats['matched_files'] = len(matched)
    stats['unmatched_files'] = len(baseline_files) - len(matched)
    
    print(f"\nMatched {len(matched)} files between baseline and output")
    print(f"Unmatched files: {stats['unmatched_files']}")
    
    if stats['unmatched_files'] > 0:
        unmatched = set(baseline_files) - len(matched)
        print(f"Sample unmatched files: {list(unmatched)[:5]}")
    
    # Compare each matched file
    for baseline_file, output_file in tqdm(matched, desc="Comparing files"):
        baseline_bboxes = baseline_data[baseline_file]
        output_bboxes = output_data[output_file]
        
        stats['total_baseline_bboxes'] += len(baseline_bboxes)
        
        uncovered_count = 0
        for baseline_bb in baseline_bboxes:
            if is_bbox_covered(baseline_bb, output_bboxes, overlap_threshold):
                stats['total_covered_bboxes'] += 1
            else:
                uncovered_count += 1
                stats['total_uncovered_bboxes'] += 1
        
        if uncovered_count > 0:
            stats['uncovered_by_file'][baseline_file] = uncovered_count
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Compare validation dataset with validated model output'
    )
    parser.add_argument(
        '--baseline',
        type=str,
        default=r'd:\projects\PaddleOCR\___dataset_for_val',
        help='Path to baseline validation dataset directory (contains Label.txt)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=r'd:\projects\issues\jira-515\newst_paddle\_val_output',
        help='Path to validated output directory'
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
        help='Show detailed per-file uncovered bbox counts'
    )
    
    args = parser.parse_args()
    
    # Construct path to Label.txt
    baseline_label_path = os.path.join(args.baseline, 'Label.txt')
    
    if not os.path.exists(baseline_label_path):
        print(f"Error: Baseline Label.txt not found at: {baseline_label_path}")
        sys.exit(1)
    
    print("Loading baseline dataset...")
    baseline_data = parse_label_txt(baseline_label_path)
    print(f"Loaded {len(baseline_data)} files from baseline")
    
    print("\nLoading validated output...")
    output_data = load_output_json(args.output)
    print(f"Loaded {len(output_data)} files from output")
    
    if len(baseline_data) == 0 or len(output_data) == 0:
        print("Error: No data loaded. Please check the paths.")
        sys.exit(1)
    
    print(f"\nComparing with overlap threshold: {args.overlap_threshold}")
    stats = compare_files(baseline_data, output_data, args.overlap_threshold)
    
    # Print results
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    print(f"Total baseline files: {stats['total_baseline_files']}")
    print(f"Total output files: {stats['total_output_files']}")
    print(f"Matched files: {stats['matched_files']}")
    print(f"Unmatched files: {stats['unmatched_files']}")
    print(f"Total baseline bounding boxes: {stats['total_baseline_bboxes']}")
    print(f"Covered bounding boxes: {stats['total_covered_bboxes']}")
    print(f"Uncovered bounding boxes: {stats['total_uncovered_bboxes']}")
    
    if stats['total_baseline_bboxes'] > 0:
        coverage_rate = 100.0 * stats['total_covered_bboxes'] / stats['total_baseline_bboxes']
        print(f"Coverage rate: {coverage_rate:.2f}%")
    
    if args.show_details and stats['uncovered_by_file']:
        print("\n" + "="*60)
        print("UNCOVERED BBOXES BY FILE")
        print("="*60)
        
        # Sort by uncovered count
        sorted_files = sorted(
            stats['uncovered_by_file'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for filename, count in sorted_files:
            print(f"{filename}: {count} uncovered bbox(es)")
    
    print("="*60)


if __name__ == '__main__':
    main()
