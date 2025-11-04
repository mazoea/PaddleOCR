import cv2
import numpy as np
import sys
import os
import time
# PaddlePaddle CPU optimization flags for Lambda
# These prevent segmentation faults on Lambda's CPU architecture
os.environ['FLAGS_use_mkldnn'] = 'false'  # Disable MKL-DNN to avoid CPU compatibility issues
os.environ['FLAGS_cpu_deterministic'] = 'true'  # Force deterministic behavior
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Allow duplicate library loading
os.environ['OMP_NUM_THREADS'] = '1'  # Limit OpenMP threads
os.environ['MKL_NUM_THREADS'] = '1'  # Limit MKL threads
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # Limit OpenBLAS threads

from paddleocr import TextDetection

def only_textspoting(image_path, output_path=None):
    # 1. Initialize the TextDetection model
    # You can specify a different model name if needed.
    # For a list of models, refer to the PaddleOCR documentation.
    # 5. Generate output path if not provided
    if output_path is None:
        base, ext = os.path.splitext(image_path)
        output_path = f"{base}_detection_result{ext}"

    print("Initializing TextDetection model...")
    s0 = time.time()
    detector = TextDetection(device='cpu', model_dir="./PP-OCRv5_mobile_det_infer_1st", model_name="PP-OCRv5_mobile_det", limit_side_len=64, limit_type='min',
                             thresh=0.3, box_thresh=0.6, unclip_ratio=1.5, cpu_threads=1, enable_mkldnn=False, mkldnn_cache_capacity=0)
    print("TextDetection model initialized in {:.3f} seconds".format(time.time() - s0))

    # 2. Read the image using OpenCV
    # The detector expects a BGR numpy array
    print(f"Reading image: {image_path}")
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    max_side_len = 1280
    h, w, _ = img.shape
    if max(h, w) > max_side_len:
        ratio = max_side_len / max(h, w)
        img = cv2.resize(img, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_AREA)
        # print(f"Resized image to {img.shape[1]}x{img.shape[0]} (width x height)")

    print(f"Image size: {img.shape[1]}x{img.shape[0]} (width x height)")

    # 3. Perform text detection
    # The result is a dictionary containing the detected polygons ('dt_polys')
    # and their scores ('dt_scores').
    print("Performing text detection...")
    s1 = time.time()
    results = detector.predict(img)
    print("Text detection completed in {:.3f} seconds".format(time.time() - s1))
    print("Text detection whole: {:.3f} seconds".format(time.time() - s0))

    detected_regions = results[0]
    print(f"Detected {len(detected_regions['dt_polys'])} text regions")

    # 4. Draw polygons around each detected text region
    output_image = img.copy()

    for i, bbox in enumerate(detected_regions["dt_polys"], 1):
        # Convert bbox to numpy array
        box = np.reshape(np.array(bbox), [-1, 1, 2]).astype(np.int64)
        # Draw blue polygon (BGR format, so blue is (255, 0, 0))
        output_image = cv2.polylines(np.array(output_image), [box], True, (255, 0, 0), 2)

    # 6. Save the result image
    cv2.imwrite(output_path, output_image)
    print(f"\nResult saved to: {output_path}")

    return output_path, detected_regions


def main():
    """Main function to handle command line arguments."""
    if len(sys.argv) < 2:
        print("Usage: python ts_v3.py <image_path> [output_path]")
        print("\nExample:")
        print("  python ts_v3.py advent.44-000001.png")
        print("  python ts_v3.py advent.44-000001.png detection_result.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Check if input file exists
    if not os.path.exists(image_path):
        print(f"Error: Input file '{image_path}' does not exist!")
        sys.exit(1)
    
    try:
        only_textspoting(image_path, output_path)
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()