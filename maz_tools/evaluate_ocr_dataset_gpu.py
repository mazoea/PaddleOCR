#!/usr/bin/env python3
"""
Evaluate OCR model performance on a dataset with ground truth labels using GPU.

This script:
1. Loads a dataset JSON file with image paths and ground truth text (gt)
2. Performs OCR on batches of images using PaddleOCR recognition model with GPU (CUDA)
3. Compares OCR results with ground truth
4. Calculates accuracy metrics
5. Processes images in batches for better GPU utilization

Usage:
    python evaluate_ocr_dataset_gpu.py --dataset <path-to-dataset.json> --model-dir <path-to-model> --batch-size <batch-size>
    
    Options:
        --dataset PATH          Path to dataset JSON file (default: d:\projects\dataset-words-OCR\__dataset.words.IBedits.json)
        --model-dir PATH        Path to OCR recognition model directory
        --batch-size NUM        Batch size for processing (default: 8)
        --output PATH           Path to save results JSON (optional)
        --show-errors           Show detailed errors for mismatches
        --limit NUM             Limit number of images to process (for testing)
    
    Examples:
        # Run with batch size 8 (default)
        python evaluate_ocr_dataset_gpu.py --dataset dataset.json --model-dir ./PP-OCRv5_mobile_rec
        
        # Use larger batch size
        python evaluate_ocr_dataset_gpu.py --dataset dataset.json --model-dir ./PP-OCRv5_mobile_rec --batch-size 16
        
        # Show detailed errors
        python evaluate_ocr_dataset_gpu.py --dataset dataset.json --model-dir ./PP-OCRv5_mobile_rec --show-errors
        
        # Save results to file
        python evaluate_ocr_dataset_gpu.py --dataset dataset.json --model-dir ./PP-OCRv5_mobile_rec --output results.json

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
from pathlib import Path
from typing import Dict, List, Tuple
import traceback

from tqdm import tqdm

# Import PaddleOCR
try:
    from paddleocr import TextRecognition
except ImportError:
    print("Error: paddleocr not found. Please install it first.")
    print("  pip install paddleocr")
    sys.exit(1)


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


def resolve_image_path(image_rel_path: str, dataset_dir: str) -> str:
    """
    Resolve image path relative to dataset directory.
    
    Args:
        image_rel_path: Relative or absolute image path
        dataset_dir: Base directory for resolving relative paths
        
    Returns:
        Absolute path to image
    """
    if os.path.isabs(image_rel_path):
        return image_rel_path
    else:
        return os.path.join(dataset_dir, image_rel_path)


def process_batch(batch_items: List[Dict], model: TextRecognition, dataset_dir: str) -> List[Dict]:
    """
    Process a batch of images with OCR.
    
    Args:
        batch_items: List of dataset items to process
        model: TextRecognition model instance
        dataset_dir: Base directory for resolving relative image paths
        
    Returns:
        List of results for each image in the batch
    """
    results = []
    
    # Prepare batch data
    image_paths = []
    ground_truths = []
    valid_indices = []
    
    for i, item in enumerate(batch_items):
        # Get image path and ground truth
        image_rel_path = item.get('id', '')
        ground_truth = item.get('gt', item.get('text', ''))
        
        if not image_rel_path or not ground_truth:
            results.append({
                'image': image_rel_path,
                'error': 'Missing image path or ground truth',
                'success': False,
                'exact_match': False,
                'correct_chars': 0,
                'total_chars': len(ground_truth) if ground_truth else 0
            })
            continue
        
        # Resolve full image path
        image_path = resolve_image_path(image_rel_path, dataset_dir)
        
        if not os.path.exists(image_path):
            results.append({
                'image': image_rel_path,
                'error': f'Image not found: {image_path}',
                'success': False,
                'exact_match': False,
                'correct_chars': 0,
                'total_chars': len(ground_truth)
            })
            continue
        
        image_paths.append(image_path)
        ground_truths.append(ground_truth)
        valid_indices.append(i)
    
    # Perform batch OCR if we have valid images
    if image_paths:
        try:
            # Run OCR on batch
            outputs = model.predict(input=image_paths, batch_size=len(image_paths))
            
            # Process outputs
            for idx, output_item in enumerate(outputs):
                original_idx = valid_indices[idx]
                item = batch_items[original_idx]
                ground_truth = ground_truths[idx]
                image_rel_path = item.get('id', '')
                
                predicted_text = output_item.get('rec_text', '')
                confidence = output_item.get('rec_score', 0.0)
                
                # Compare with ground truth
                exact_match = predicted_text == ground_truth
                correct_chars, total_chars = calculate_char_accuracy(predicted_text, ground_truth)
                
                results.append({
                    'image': image_rel_path,
                    'ground_truth': ground_truth,
                    'predicted': predicted_text,
                    'confidence': confidence,
                    'exact_match': exact_match,
                    'correct_chars': correct_chars,
                    'total_chars': total_chars,
                    'success': True
                })
            
        except Exception as e:
            error_msg = f"Batch processing error: {str(e)}\n{traceback.format_exc()}"
            print(f"\nError processing batch: {error_msg}")
            
            # Add error results for all images in batch
            for idx in valid_indices:
                if idx >= len(results):
                    item = batch_items[idx]
                    results.append({
                        'image': item.get('image', 'unknown'),
                        'error': error_msg,
                        'success': False,
                        'exact_match': False,
                        'correct_chars': 0,
                        'total_chars': len(item.get('gt', item.get('text', '')))
                    })
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate OCR model performance on a dataset using GPU with batch processing',
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
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for processing (default: 8)'
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
    print(f"Batch size: {args.batch_size}")
    print(f"Device: GPU (CUDA)")
    print()
    
    # Initialize model with GPU
    print("Initializing OCR model with GPU...")
    model_name = "PP-OCRv5_server_rec" if 'server' in args.model_dir else "PP-OCRv5_mobile_rec"
    
    try:
        model = TextRecognition(
            model_name=model_name,
            device='gpu',  # Use GPU
            model_dir=args.model_dir
        )
        print("Model loaded successfully on GPU")
    except Exception as e:
        print(f"Error initializing model on GPU: {e}")
        print("Make sure CUDA is installed and paddlepaddle-gpu is installed correctly.")
        sys.exit(1)
    
    # Process images in batches
    all_results = []
    total = 0
    exact_matches = 0
    total_chars = 0
    correct_chars = 0
    errors = []
    
    start_time = time.time()
    
    # Create batches
    num_batches = (len(dataset) + args.batch_size - 1) // args.batch_size
    
    with tqdm(total=len(dataset), desc="Processing images") as pbar:
        for batch_idx in range(num_batches):
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, len(dataset))
            batch_items = dataset[start_idx:end_idx]
            
            # Process batch
            batch_results = process_batch(batch_items, model, dataset_dir)
            all_results.extend(batch_results)
            
            # Update statistics
            for result in batch_results:
                if result.get('success', False):
                    total += 1
                    if result.get('exact_match', False):
                        exact_matches += 1
                    correct_chars += result.get('correct_chars', 0)
                    total_chars += result.get('total_chars', 0)
                    
                    if not result.get('exact_match', False) and args.show_errors:
                        errors.append({
                            'image': result['image'],
                            'ground_truth': result['ground_truth'],
                            'predicted': result['predicted'],
                            'confidence': result['confidence']
                        })
            
            pbar.update(len(batch_items))
    
    elapsed_time = time.time() - start_time
    
    # Calculate final statistics
    exact_match_rate = 100.0 * exact_matches / total if total > 0 else 0.0
    char_accuracy = 100.0 * correct_chars / total_chars if total_chars > 0 else 0.0
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS (GPU)")
    print("="*60)
    print(f"Total images processed: {total}")
    print(f"Exact matches: {exact_matches} ({exact_match_rate:.2f}%)")
    print(f"Character accuracy: {correct_chars}/{total_chars} ({char_accuracy:.2f}%)")
    print(f"Processing time: {elapsed_time:.2f}s")
    print(f"Average time per image: {elapsed_time/total:.3f}s" if total > 0 else "")
    print(f"Average time per batch: {elapsed_time/num_batches:.3f}s")
    print("="*60)
    
    # Show errors if requested
    if args.show_errors and errors:
        print(f"\nShowing {len(errors)} mismatches:")
        print("-" * 60)
        for i, error in enumerate(errors[:20], 1):  # Show first 20
            print(f"\n{i}. Image: {error['image']}")
            print(f"   Ground Truth: '{error['ground_truth']}'")
            print(f"   Predicted:    '{error['predicted']}' (conf: {error['confidence']:.3f})")
        
        if len(errors) > 20:
            print(f"\n... and {len(errors) - 20} more errors")
    
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
                'avg_time_per_image': elapsed_time / total if total > 0 else 0,
                'avg_time_per_batch': elapsed_time / num_batches,
                'batch_size': args.batch_size,
                'device': 'GPU (CUDA)'
            },
            'results': all_results
        }
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nDetailed results saved to: {args.output}")
    
    print()


if __name__ == '__main__':
    main()
