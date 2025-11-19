#!/usr/bin/env python3
"""
OCR images in a folder and save results to JSON file using GPU batch processing.

This script:
1. Scans a folder for images
2. Performs OCR on batches of images using PaddleOCR recognition model with GPU (CUDA)
3. Saves results to a JSON file with id, ocred text, confidence, and gt fields

Usage:
    python ocr_folder_to_json_gpu.py --input-dir <folder-with-images> --output <output.json> --model-dir <path-to-model>
    
    Options:
        --input-dir PATH        Path to folder containing images (required)
        --output PATH           Path to save results JSON (required)
        --model-dir PATH        Path to OCR recognition model directory (required)
        --batch-size NUM        Batch size for processing (default: 8)
        --extensions EXT        Comma-separated image extensions (default: jpg,jpeg,png,bmp,tiff)
        --recursive             Search images recursively in subdirectories
        --limit NUM             Limit number of images to process (for testing)
    
    Examples:
        # Basic usage
        python ocr_folder_to_json_gpu.py --input-dir ./images --output results.json --model-dir ./PP-OCRv5_mobile_rec
        
        # Use larger batch size
        python ocr_folder_to_json_gpu.py --input-dir ./images --output results.json --model-dir ./PP-OCRv5_mobile_rec --batch-size 16
        
        # Process subdirectories recursively
        python ocr_folder_to_json_gpu.py --input-dir ./images --output results.json --model-dir ./PP-OCRv5_mobile_rec --recursive
        
        # Limit to first 100 images for testing
        python ocr_folder_to_json_gpu.py --input-dir ./images --output results.json --model-dir ./PP-OCRv5_mobile_rec --limit 100

Output JSON format:
    [
        {
            "id": "image1.jpg",
            "ocred": "recognized text",
            "conf": 0.9876,
            "gt": "recognized text"
        },
        ...
    ]
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict
import traceback

from tqdm import tqdm

# Import PaddleOCR
try:
    from paddleocr import TextRecognition
except ImportError:
    print("Error: paddleocr not found. Please install it first.")
    print("  pip install paddleocr")
    sys.exit(1)


def find_images(input_dir: str, extensions: List[str], recursive: bool = False) -> List[str]:
    """
    Find all images in the input directory.
    
    Args:
        input_dir: Directory to search for images
        extensions: List of valid image extensions (without dot)
        recursive: Whether to search recursively in subdirectories
        
    Returns:
        List of image file paths
    """
    image_files = []
    extensions_set = set(ext.lower() for ext in extensions)
    
    if recursive:
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower().lstrip('.')
                if ext in extensions_set:
                    image_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(input_dir):
            file_path = os.path.join(input_dir, file)
            if os.path.isfile(file_path):
                ext = os.path.splitext(file)[1].lower().lstrip('.')
                if ext in extensions_set:
                    image_files.append(file_path)
    
    return sorted(image_files)


def process_batch(batch_paths: List[str], model: TextRecognition) -> List[Dict]:
    """
    Process a batch of images with OCR.
    
    Args:
        batch_paths: List of image file paths to process
        model: TextRecognition model instance
        
    Returns:
        List of results for each image in the batch
    """
    results = []
    
    if not batch_paths:
        return results
    
    try:
        # Run OCR on batch
        outputs = model.predict(input=batch_paths, batch_size=len(batch_paths))
        
        # Process outputs
        for image_path, output_item in zip(batch_paths, outputs):
            predicted_text = output_item.get('rec_text', '')
            confidence = output_item.get('rec_score', 0.0)
            
            # Get filename (id)
            filename = os.path.basename(image_path)
            
            result = {
                'id': filename,
                'ocred': predicted_text,
                'conf': float(confidence),
                'gt': predicted_text  # gt is same as ocred as requested
            }
            
            results.append(result)
        
    except Exception as e:
        error_msg = f"Batch processing error: {str(e)}\n{traceback.format_exc()}"
        print(f"\nError processing batch: {error_msg}")
        
        # Add error results for all images in batch
        for image_path in batch_paths:
            filename = os.path.basename(image_path)
            results.append({
                'id': filename,
                'ocred': '',
                'conf': 0.0,
                'gt': '',
                'error': error_msg
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='OCR images in a folder and save results to JSON using GPU batch processing',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Path to folder containing images'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save results JSON file'
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
        '--extensions',
        type=str,
        default='jpg,jpeg,png,bmp,tiff',
        help='Comma-separated image extensions (default: jpg,jpeg,png,bmp,tiff)'
    )
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Search images recursively in subdirectories'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of images to process (for testing)'
    )
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input path is not a directory: {args.input_dir}")
        sys.exit(1)
    
    # Check if model directory exists
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory not found: {args.model_dir}")
        sys.exit(1)
    
    # Parse extensions
    extensions = [ext.strip() for ext in args.extensions.split(',')]
    
    # Find images
    print(f"Scanning for images in: {args.input_dir}")
    print(f"Extensions: {', '.join(extensions)}")
    print(f"Recursive: {args.recursive}")
    print()
    
    image_files = find_images(args.input_dir, extensions, args.recursive)
    
    if not image_files:
        print(f"Error: No images found in {args.input_dir}")
        sys.exit(1)
    
    if args.limit:
        image_files = image_files[:args.limit]
        print(f"Limited to first {args.limit} images")
    
    print(f"Found {len(image_files)} images")
    print(f"Model directory: {args.model_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: GPU (CUDA)")
    print(f"Output: {args.output}")
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
    
    start_time = time.time()
    
    # Create batches
    num_batches = (len(image_files) + args.batch_size - 1) // args.batch_size
    
    with tqdm(total=len(image_files), desc="Processing images") as pbar:
        for batch_idx in range(num_batches):
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, len(image_files))
            batch_paths = image_files[start_idx:end_idx]
            
            # Process batch
            batch_results = process_batch(batch_paths, model)
            all_results.extend(batch_results)
            
            pbar.update(len(batch_paths))
    
    elapsed_time = time.time() - start_time
    
    # Print statistics
    successful = sum(1 for r in all_results if 'error' not in r)
    failed = len(all_results) - successful
    
    print("\n" + "="*60)
    print("OCR PROCESSING COMPLETE")
    print("="*60)
    print(f"Total images processed: {len(all_results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Processing time: {elapsed_time:.2f}s")
    print(f"Average time per image: {elapsed_time/len(all_results):.3f}s" if all_results else "")
    print(f"Average time per batch: {elapsed_time/num_batches:.3f}s")
    print("="*60)
    
    # Save results to JSON
    print(f"\nSaving results to: {args.output}")
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved successfully!")
    print(f"\nOutput format:")
    print(f"  - id: filename")
    print(f"  - ocred: recognized text")
    print(f"  - conf: confidence score (0.0 - 1.0)")
    print(f"  - gt: same as ocred (as requested)")
    
    # Show sample results
    if all_results:
        print(f"\nSample results (first 3):")
        for i, result in enumerate(all_results[:3], 1):
            print(f"\n{i}. {result['id']}")
            print(f"   OCRed: '{result['ocred']}'")
            print(f"   Confidence: {result['conf']:.4f}")
            if 'error' in result:
                print(f"   Error: {result['error'][:100]}...")
    
    print()


if __name__ == '__main__':
    main()
