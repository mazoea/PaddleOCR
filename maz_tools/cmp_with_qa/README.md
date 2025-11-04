# OCR Results Comparison Script

This directory contains a Python script that compares two different OCR processing results from the same dataset.

## Files

- **compare_results.py** - Main comparison script
- **comparison_results.csv** - Detailed comparison results (generated after running the script)

## Overview

The script compares two OCR processing results:

1. **Result 1** (qa_output): Contains bounding boxes in simple JSON format
   - Location: `D:\projects\issues\jira-515\newst_paddle\qa_output\`
   - Format: `*.bbs.json` files with array of `{x, y, w, h}` objects
   - Example: `37.pdf-000001.png_detected_lambda_2.340_1078.png.bbs.json`

2. **Result 2** (qa_dp_advent): Contains bounding boxes in nested page structure
   - Location: `D:\projects\issues\jira-515\newst_paddle\qa_dp_advent\`
   - Format: `*.raw.json` files with nested structure: `pages[0].blocks[].lines[].words[].bbox`
   - Example: `37.pdf-000001.png.raw.json`

## How It Works

The script performs the following steps:

1. **File Matching**: Matches files from both directories by prefix (e.g., `37.pdf-000001.png`)
   
2. **Coordinate Normalization**: 
   - Extracts image size from Result 2 (stored in `pages[0].bbox`)
   - Infers image size from Result 1 (from maximum bbox coordinates)
   - Normalizes all bounding boxes to the same coordinate system

3. **Box Comparison**:
   - Calculates IoU (Intersection over Union) between all box pairs
   - Boxes with IoU ≥ 50% are considered "matching"
   - Counts unique boxes in each result (not matched by the other)

4. **Results**:
   - Reports which result has more unique (non-overlapping) bounding boxes
   - Saves detailed comparison to CSV file
   - Provides summary statistics

## Usage

```bash
python compare_results.py
```

## Output

The script generates:

1. **Console Output**: 
   - Progress for each file pair
   - Summary statistics
   - Top 10 cases with most unique boxes per result

2. **comparison_results.csv**: 
   - Detailed results for each comparison
   - Columns: prefix, total_boxes_result1, total_boxes_result2, common_boxes, unique_result1, unique_result2, winner, difference

## Results Summary

Based on the comparison of 481 file pairs:

- **Result 2 (qa_dp_advent)** consistently has more unique boxes than Result 1 (qa_output)
- Result 2 won in all 481 comparisons (100%)
- Result 1 won in 0 comparisons (0%)
- No ties

This indicates that the qa_dp_advent processing detects significantly more word bounding boxes that are not found by the qa_output processing.

## Key Findings

**Top cases where Result 2 found the most additional boxes:**
1. advent.jira-318.pdf-000003.png: 1230 additional unique boxes
2. advent.jira-318.pdf-000018.png: 1194 additional unique boxes
3. advent.jira-318.pdf-000024.png: 1170 additional unique boxes
4. advent.551.pdf-000001.png: 1004 additional unique boxes
5. advent.742.pdf-000001.png: 954 additional unique boxes

## Requirements

- Python 3.6+
- No external dependencies (uses only standard library)

## Notes

- The overlap threshold is set to 50% (IoU ≥ 0.5) to consider boxes as matching
- Coordinate normalization handles different image sizes between results
- The script handles cases where files don't match or have empty results gracefully
