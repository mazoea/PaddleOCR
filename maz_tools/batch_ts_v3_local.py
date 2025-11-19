#!/usr/bin/env python3
"""
    Batch process images in folder by calling ts_v3.py for each image.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_image_files(folder_path):
    """
    Get all image files from the specified folder.
    
    Args:
        folder_path: Path to the folder containing images
    
    Returns:
        List of image file paths
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    # Common image extensions
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp'}
    
    image_files = []
    for file_path in folder.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)
    
    return sorted(image_files)


def process_file(image_path, output_dir, model_dir, heat_map, bbs, words_output):
    """
    Process a single file using invoke_lambda.py
    
    Args:
        image_path: Path to the image file
        output_dir: Output directory path
    
    Returns:
        Tuple of (image_path, success, output)
    """
    try:
        # Run invoke_lambda.py with the image file
        cmd = [sys.executable, 'ts_v3.py', f'--input={str(image_path)}', f'--output={output_dir}',
               f'--model_dir={model_dir}', f'--heat_map={str(heat_map)}', f'--bbs={str(bbs)}', f'--words_output={words_output}']

        print(f'[START] Processing: {" ".join(cmd)}')
        
        # Run the process and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        
        # Print stdout in real-time
        output = result.stdout
        if output:
            print(output, end="")
        
        # Print stderr if there's an error
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        success = result.returncode == 0
        
        #if success:
        #    print(f"[SUCCESS] {image_path.name}")
        #else:
        #    print(f"[FAILED] {image_path.name} (exit code: {result.returncode})")
        
        return image_path, success, output
        
    except Exception as e:
        error_msg = f"[ERROR] {image_path.name}: {str(e)}"
        print(error_msg, file=sys.stderr)
        return image_path, False, str(e)


def process_chunk(chunk, output_dir, chunk_idx, total_chunks, model_dir, heat_map, bbs, words_output):
    """
    Process a chunk of files in parallel
    
    Args:
        chunk: List of image paths to process
        output_dir: Output directory path
        chunk_idx: Current chunk index
        total_chunks: Total number of chunks
    
    Returns:
        List of results
    """
    #print(f"\n{'='*80}")
    #print(f"Processing Chunk {chunk_idx}/{total_chunks} ({len(chunk)} files)")
    #print(f"{'='*80}\n")
    
    results = []
    
    # Process files in the chunk in parallel
    with ThreadPoolExecutor(max_workers=len(chunk)) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_file, img_path, output_dir, model_dir, heat_map, bbs, words_output): img_path
            for img_path in chunk
        }
        
        # Collect results as they complete
        for future in as_completed(futures):
            img_path = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"[EXCEPTION] {img_path.name}: {str(e)}", file=sys.stderr)
                results.append((img_path, False, str(e)))
    
    return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Run command for textspotting in parallel on images in a folder.')
    parser.add_argument('--input', help='input dir',
                        type=str, required=True)
    parser.add_argument('--output', help='OutputPath',
                        type=str, required=False, default=None)
    parser.add_argument('--model_dir', help='Model dir where model with name PP-OCRv5_mobile_det will be stored',
                        type=str, required=True)
    parser.add_argument('--heat_map', help='if 1 print heat map image for debugging',
                        type=int, required=False, default=0)
    parser.add_argument('--bbs', help='if 1 print bounding boxes image for debugging',
                        type=int, required=False, default=0)
    parser.add_argument('--words_output', help='Create image of words and stored into words_output folder',
                        type=str, required=False, default=None)
    parser.add_argument('--threads', help='Number of threads',
                        type=int, required=False, default=8)

    flags = parser.parse_args()

    input_folder = flags.input
    output_dir = flags.output
    model_dir = flags.model_dir
    heat_map = flags.heat_map
    bbs = flags.bbs
    chunk_size = flags.threads
    words_output = flags.words_output

    if chunk_size < 1:
        print("Error: chunk_size must be at least 1")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Input folder: {input_folder}")
    print(f"Chunk size: {chunk_size}")
    print(f"Output directory: {output_dir}")
    
    # Get all image files
    try:
        image_files = get_image_files(input_folder)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)
    
    if not image_files:
        print(f"No image files found in {input_folder}")
        sys.exit(0)
    
    print(f"\nFound {len(image_files)} image files")
    
    # Split files into chunks
    chunks = [image_files[i:i + chunk_size] for i in range(0, len(image_files), chunk_size)]
    total_chunks = len(chunks)
    
    print(f"Processing in {total_chunks} chunks of up to {chunk_size} files each\n")
    
    # Process each chunk
    all_results = []
    for idx, chunk in enumerate(chunks, 1):
        chunk_results = process_chunk(chunk, output_dir, idx, total_chunks, model_dir, heat_map, bbs, words_output)
        all_results.extend(chunk_results)
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    successful = sum(1 for _, success, _ in all_results if success)
    failed = len(all_results) - successful
    
    print(f"Total files: {len(all_results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\nFailed files:")
        for img_path, success, output in all_results:
            if not success:
                print(f"  - {img_path.name}")
    
    print(f"{'='*80}\n")
    
    # Exit with error code if any failed
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
