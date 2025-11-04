#!/usr/bin/env python3
"""
Scale all images in a directory so that the maximum dimension (width or height)
is 1280px while maintaining the original aspect ratio.

Images are resized in-place (original files are replaced).
"""

import os
from pathlib import Path
from PIL import Image

# Configuration
SOURCE_DIR = r'.'
MAX_SIZE = 1280
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}


def resize_image(image_path: str, max_size: int = 1280) -> tuple:
    """
    Resize an image so that its maximum dimension is max_size,
    maintaining aspect ratio.
    
    Returns: (original_size, new_size, was_resized)
    """
    try:
        # Open image
        img = Image.open(image_path)
        original_size = img.size  # (width, height)
        
        # Get current dimensions
        width, height = img.size
        max_dim = max(width, height)
        
        # Check if resizing is needed
        if max_dim <= max_size:
            print(f"  Skipped (already <= {max_size}px): {width}x{height}")
            return original_size, original_size, False
        
        # Calculate new dimensions maintaining aspect ratio
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        
        new_size = (new_width, new_height)
        
        # Resize image using high-quality resampling
        resized_img = img.resize(new_size, Image.LANCZOS)
        
        # Save back to the same file
        resized_img.save(image_path)
        
        print(f"  Resized: {width}x{height} -> {new_width}x{new_height}")
        
        return original_size, new_size, True
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return None, None, False


def main():
    """Main function to resize all images in the directory."""
    print("=" * 80)
    print(f"Image Resizing Script - Max dimension: {MAX_SIZE}px")
    print("=" * 80)
    print(f"Directory: {SOURCE_DIR}")
    print()
    
    # Check if directory exists
    if not os.path.exists(SOURCE_DIR):
        print(f"ERROR: Directory not found: {SOURCE_DIR}")
        return
    
    # Get all image files
    image_files = []
    for filename in os.listdir(SOURCE_DIR):
        filepath = os.path.join(SOURCE_DIR, filename)
        if os.path.isfile(filepath):
            ext = os.path.splitext(filename)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                image_files.append((filename, filepath))
    
    print(f"Found {len(image_files)} image files")
    print()
    
    if not image_files:
        print("No image files found!")
        return
    
    # Ask for confirmation
    print("WARNING: This will modify images in-place (original files will be replaced)")
    response = input("Do you want to continue? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("Operation cancelled.")
        return
    
    print()
    print("Processing images...")
    print()
    
    # Process each image
    resized_count = 0
    skipped_count = 0
    error_count = 0
    
    for filename, filepath in image_files:
        print(f"Processing: {filename}")
        original_size, new_size, was_resized = resize_image(filepath, MAX_SIZE)
        
        if original_size is None:
            error_count += 1
        elif was_resized:
            resized_count += 1
        else:
            skipped_count += 1
        
        print()
    
    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total images processed: {len(image_files)}")
    print(f"Resized: {resized_count}")
    print(f"Skipped (already small enough): {skipped_count}")
    print(f"Errors: {error_count}")
    print()
    print("Done!")


if __name__ == '__main__':
    main()
