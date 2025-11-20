#!/usr/bin/env python3
"""
Evaluate OCR model performance on a dataset with ground truth labels.

This script:
1. Loads a dataset JSON file with image paths and ground truth text (gt)
2. Performs OCR on each image using PaddleOCR recognition model
3. Compares OCR results with ground truth
4. Calculates accuracy metrics
5. Uses multi-threading for parallel processing

Usage:
    python evaluate_ocr_dataset.py --dataset <path-to-dataset.json> --model-dir <path-to-model> --threads <num-threads>
    
    Options:
        --dataset PATH          Path to dataset JSON file (default: d:\projects\dataset-words-OCR\__dataset.words.IBedits.json)
        --model-dir PATH        Path to OCR recognition model directory
        --threads NUM           Number of parallel threads (default: 4)
        --output PATH           Path to save results JSON (optional)
        --show-errors           Show detailed errors for mismatches
    
    Examples:
        # Run with 8 threads
        python evaluate_ocr_dataset.py --dataset dataset.json --model-dir ./PP-OCRv5_mobile_rec --threads 8
        
        # Show detailed errors
        python evaluate_ocr_dataset.py --dataset dataset.json --model-dir ./PP-OCRv5_mobile_rec --show-errors
        
        # Save results to file
        python evaluate_ocr_dataset.py --dataset dataset.json --model-dir ./PP-OCRv5_mobile_rec --output results.json

Output:
    - Total images processed
    - Exact matches (OCR text exactly equals ground truth)
    - Character accuracy (percentage of correct characters)
    - Word accuracy (percentage of correct words)
    - Per-image results (with --show-errors)
"""

import json
import os
import sys
import time
import argparse
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import traceback

from tqdm import tqdm

# Import PaddleOCR
try:
    from paddleocr import TextRecognition
except ImportError:
    print("Error: paddleocr not found. Please install it first.")
    print("  pip install paddleocr")
    sys.exit(1)


# Global variables for thread-safe operations
stats_lock = Lock()
global_stats = {
    'total': 0,
    'exact_matches': 0,
    'total_chars': 0,
    'correct_chars': 0,
    'errors': []
}


def load_dataset(dataset_path: str) -> List[Dict]:
    """
    Load dataset from JSON file.
    
    Expected format:
    [
        {
            "image": "path/to/image.png",
            "gt": "ground truth text",
            ...
        },
        ...
    ]
    
    Or:
    {
        "data": [
            {"image": "path/to/image.png", "gt": "ground truth text"},
            ...
        ]
    }
    
    Args:
        dataset_path: Path to dataset JSON file
        
    Returns:
        List of dataset items
    """
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        if 'data' in data:
            return data['data']
        elif 'images' in data:
            return data['images']
        else:
            # Try to convert dict to list
            return [data]
    else:
        raise ValueError(f"Unsupported dataset format in {dataset_path}")


def calculate_char_accuracy(predicted: str, ground_truth: str) -> Tuple[int, int]:
    """
    Calculate character-level accuracy using simple matching.
    
    Args:
        predicted: Predicted text
        ground_truth: Ground truth text
        
    Returns:
        Tuple of (correct_chars, total_chars)
    """
    # Simple character matching
    correct = 0
    total = len(ground_truth)
    
    for i in range(min(len(predicted), len(ground_truth))):
        if predicted[i] == ground_truth[i]:
            correct += 1
    
    return correct, total


def perform_ocr(image_path: str, model: TextRecognition) -> Tuple[str, float]:
    """
    Perform OCR on a single image.
    
    Args:
        image_path: Path to image file
        model: TextRecognition model instance
        
    Returns:
        Tuple of (recognized_text, confidence_score)
    """
    try:
        output = model.predict(input=image_path, batch_size=1)
        
        if output and len(output) > 0:
            result = output[0]
            text = result.get('rec_text', '')
            score = result.get('rec_score', 0.0)
            return text, score
        else:
            return '', 0.0
            
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return '', 0.0


def process_single_image(item: Dict, model_dir: str, dataset_dir: str, show_errors: bool) -> Dict:
    """
    Process a single image: perform OCR and compare with ground truth.
    
    Args:
        item: Dataset item with 'image' and 'gt' keys
        model_dir: Path to OCR model directory
        dataset_dir: Base directory for resolving relative image paths
        show_errors: Whether to track detailed errors
        
    Returns:
        Dictionary with results
    """
    try:
        # Get image path and ground truth
        image_rel_path = item.get('id','')
        ground_truth = item.get('gt', item.get('text', ''))
        
        if not image_rel_path or not ground_truth:
            return {
                'image': image_rel_path,
                'error': 'Missing image path or ground truth',
                'success': False
            }

        is_pad = "pad" in image_rel_path
        if is_pad:
            return {
                'image': image_rel_path,
                'error': 'Do not work with PAD images',
                'success': False
            }

        
        # Resolve full image path
        if os.path.isabs(image_rel_path):
            image_path = image_rel_path
        else:
            image_path = os.path.join(dataset_dir, image_rel_path)
        
        if not os.path.exists(image_path):
            return {
                'image': image_rel_path,
                'error': f'Image not found: {image_path}',
                'success': False
            }
        
        # Initialize model (each thread gets its own model instance)
        model_name = "PP-OCRv5_mobile_rec"
        if "PP-OCRv5_mobile_rec" in model_dir:
            model_name = "PP-OCRv5_mobile_rec"
        if "PP-OCRv5_server_rec" in model_dir:
            model_name = "PP-OCRv5_server_rec"
        if  "en_PP-OCRv5_mobile_rec" in model_dir:
            model_name = "en_PP-OCRv5_mobile_rec"
        if "latin_PP-OCRv5_mobile_rec" in model_dir:
            model_name = "latin_PP-OCRv5_mobile_rec"
        model = TextRecognition(
            model_name=model_name,
            device='cpu',
            cpu_threads=1,
            enable_mkldnn=False,
            mkldnn_cache_capacity=0,
            model_dir=model_dir
        )
        
        # Perform OCR
        predicted_text, confidence = perform_ocr(image_path, model)
        
        # Compare with ground truth
        exact_match = predicted_text == ground_truth
        correct_chars, total_chars = calculate_char_accuracy(predicted_text, ground_truth)
        
        # Update global statistics (thread-safe)
        with stats_lock:
            global_stats['total'] += 1
            if exact_match:
                global_stats['exact_matches'] += 1
            global_stats['total_chars'] += total_chars
            global_stats['correct_chars'] += correct_chars
            
            if not exact_match and show_errors:
                global_stats['errors'].append({
                    'image': image_rel_path,
                    'ground_truth': ground_truth,
                    'predicted': predicted_text,
                    'confidence': confidence
                })
        
        return {
            'image': image_rel_path,
            'ground_truth': ground_truth,
            'predicted': predicted_text,
            'confidence': confidence,
            'exact_match': exact_match,
            'correct_chars': correct_chars,
            'total_chars': total_chars,
            'success': True
        }
        
    except Exception as e:
        error_msg = f"Exception: {str(e)}\n{traceback.format_exc()}"
        return {
            'image': item.get('image', 'unknown'),
            'error': error_msg,
            'success': False
        }


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate OCR model performance on a dataset with multi-threading',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=r'd:\projects\dataset-words-OCR\__dataset.words.IBedits.json',
        help='Path to dataset JSON file'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        required=True,
        help='Path to OCR recognition model directory'
    )
    parser.add_argument(
        '--threads',
        type=int,
        default=4,
        help='Number of parallel threads (default: 4)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Path to save detailed results JSON (optional)'
    )
    parser.add_argument(
        '--show-errors',
        action='store_true',
        help='Show detailed errors for mismatches'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of images to process (for testing)'
    )
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset file not found: {args.dataset}")
        sys.exit(1)
    
    # Check if model directory exists
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory not found: {args.model_dir}")
        sys.exit(1)
    
    # Get dataset directory for resolving relative paths
    dataset_dir = os.path.dirname(os.path.abspath(args.dataset))
    
    print(f"Loading dataset from: {args.dataset}")
    dataset = load_dataset(args.dataset)

    # remove PAD images
    dataset = [item for item in dataset if "pad" not in item.get('id','pad')]
    
    if args.limit:
        dataset = dataset[:args.limit]
        print(f"Limited to first {args.limit} images")
    
    print(f"Dataset loaded: {len(dataset)} images")
    print(f"Model directory: {args.model_dir}")
    print(f"Using {args.threads} threads")
    print()
    
    # Reset global stats
    global_stats['total'] = 0
    global_stats['exact_matches'] = 0
    global_stats['total_chars'] = 0
    global_stats['correct_chars'] = 0
    global_stats['errors'] = []
    
    # Process images with multi-threading
    results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_single_image, item, args.model_dir, dataset_dir, args.show_errors): item
            for item in dataset
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(dataset), desc="Processing images") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                pbar.update(1)
    
    elapsed_time = time.time() - start_time
    
    # Calculate final statistics
    total = global_stats['total']
    exact_matches = global_stats['exact_matches']
    total_chars = global_stats['total_chars']
    correct_chars = global_stats['correct_chars']
    
    exact_match_rate = 100.0 * exact_matches / total if total > 0 else 0.0
    char_accuracy = 100.0 * correct_chars / total_chars if total_chars > 0 else 0.0
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Total images processed: {total}")
    print(f"Exact matches: {exact_matches} ({exact_match_rate:.2f}%)")
    print(f"Character accuracy: {correct_chars}/{total_chars} ({char_accuracy:.2f}%)")
    print(f"Processing time: {elapsed_time:.2f}s")
    print(f"Average time per image: {elapsed_time/total:.3f}s" if total > 0 else "")
    print("="*60)
    
    # Show errors if requested
    if args.show_errors and global_stats['errors']:
        print(f"\nShowing {len(global_stats['errors'])} mismatches:")
        print("-" * 60)
        for i, error in enumerate(global_stats['errors'][:20], 1):  # Show first 20
            print(f"\n{i}. Image: {error['image']}")
            print(f"   Ground Truth: '{error['ground_truth']}'")
            print(f"   Predicted:    '{error['predicted']}' (conf: {error['confidence']:.3f})")
        
        if len(global_stats['errors']) > 20:
            print(f"\n... and {len(global_stats['errors']) - 20} more errors")
    
    # Save detailed results if requested
    if args.output:
        output_data = {
            'summary': {
                'total': total,
                'exact_matches': exact_matches,
                'exact_match_rate': exact_match_rate,
                'correct_chars': correct_chars,
                'total_chars': total_chars,
                'char_accuracy': char_accuracy,
                'processing_time': elapsed_time,
                'avg_time_per_image': elapsed_time / total if total > 0 else 0
            },
            'results': results
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nDetailed results saved to: {args.output}")
    
    print()


if __name__ == '__main__':
    main()
