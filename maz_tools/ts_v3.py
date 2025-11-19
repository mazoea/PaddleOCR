import cv2
import numpy as np
import sys
import os
import time
import argparse
import json

# PaddlePaddle CPU optimization flags for Lambda
# These prevent segmentation faults on Lambda's CPU architecture
os.environ['FLAGS_use_mkldnn'] = 'false'  # Disable MKL-DNN to avoid CPU compatibility issues
os.environ['FLAGS_cpu_deterministic'] = 'true'  # Force deterministic behavior
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Allow duplicate library loading
os.environ['OMP_NUM_THREADS'] = '1'  # Limit OpenMP threads
os.environ['MKL_NUM_THREADS'] = '1'  # Limit MKL threads
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # Limit OpenBLAS threads

from paddleocr import TextDetection

def textspotting(image_path: str, model_dir: str, heat_map: bool, bbs: bool, output_path=None, word_output=None):
    # 1. Initialize the TextDetection model
    # You can specify a different model name if needed.
    # For a list of models, refer to the PaddleOCR documentation.
    # 5. Generate output path if not provided
    file_name = ""
    if output_path is None:
        output_path = f"{image_path}_detection.png"
    else:
        file_name = os.path.basename(image_path)
        output_path = os.path.join(output_path, f"{file_name}_detection.png")

    print("Initializing TextDetection model...")
    s0 = time.time()
    detector = TextDetection(device='cpu', model_dir=model_dir, model_name="PP-OCRv5_mobile_det", limit_side_len=64, limit_type='min',
                             thresh=0.1, box_thresh=0.3, unclip_ratio=1.5, cpu_threads=1, enable_mkldnn=False, mkldnn_cache_capacity=0)
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
    if heat_map:
        os.environ["MAZ_OUTPUT_PATH"] = output_path  # For debugging purposes
    else:
        if "MAZ_OUTPUT_PATH" in os.environ:
            del os.environ["MAZ_OUTPUT_PATH"]
    results = detector.predict(img)
    print("Text detection completed in {:.3f} seconds".format(time.time() - s1))
    print("Text detection whole: {:.3f} seconds".format(time.time() - s0))

    detected_regions = results[0]
    print(f"Detected {len(detected_regions['dt_polys'])} text regions")

    # 4. Draw polygons around each detected text region
    output_image = img.copy()

    bboxes = []
    for i, bbox in enumerate(detected_regions["dt_polys"], 1):
        # Convert bbox to numpy array
        box = np.reshape(np.array(bbox), [-1, 1, 2]).astype(np.int64)
        # Draw blue polygon (BGR format, so blue is (255, 0, 0))
        output_image = cv2.polylines(np.array(output_image), [box], True, (255, 0, 0), 2)

        box_points = box.tolist()
        x_coords = [point[0][0] for point in box_points]
        y_coords = [point[0][1] for point in box_points]

        # Find min and max for x and y
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)

        # Calculate width, height, and center coordinates
        w = x_max - x_min
        h = y_max - y_min

        # Create the desired dictionary
        result_dict = {
            "h": int(h),
            "w": int(w),
            "x": int(x_min),
            "y": int(y_min)
        }
        bboxes.append(result_dict)

        # if word_output is specified, save each detected word region
        if word_output is not None:
            word_img = img[y_min:y_max, x_min:x_max]
            os.makedirs(word_output, exist_ok=True)
            # random number to avoid overwriting
            num = np.random.randint(0, 2e6)
            img_file_name = file_name + "_"+ str(num)
            word_image_path = os.path.join(word_output, f"{img_file_name}.png")
            cv2.imwrite(word_image_path, word_img)

    # Save the result image
    cv2.imwrite(output_path, output_image)
    print(f"\nResult saved to: {output_path}")

    # Save bounding boxes if requested
    if bbs:
        bboxes_path = f"{output_path}.bbs.json"
        with open(bboxes_path, 'w') as f:
            json.dump({"bboxes": bboxes, "img_h": img.shape[0], "img_w": img.shape[1]}, f)

    return output_path, detected_regions


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(description='Run Paddle textspotting locally. Possibility to set model '
                                                 'for test and output path, where to save results.')
    parser.add_argument('--input', help='Input image',
                        type=str, required=True)
    parser.add_argument('--output', help='OutputPath',
                        type=str, required=False, default=None)
    parser.add_argument('--model_dir', help='Model dir where model with name PP-OCRv5_mobile_det will be stored',
                        type=str, required=True)
    parser.add_argument('--heat_map', help='Print heat map image for debugging',
                        type=int, required=False, default=0)
    parser.add_argument('--bbs', help='Print bounding boxes image for debugging',
                        type=int, required=False, default=0)
    parser.add_argument('--words_output', help='Create image of words and stored into words_output folder',
                        type=str, required=False, default=None)
    flags = parser.parse_args()

    image_path = flags.input
    output_path = flags.output
    model_dir = flags.model_dir
    heat_map = flags.heat_map != 0
    bbs = flags.bbs != 0
    words_output = flags.words_output
    
    # Check if input file exists
    if not os.path.exists(image_path):
        print(f"Error: Input file '{image_path}' does not exist!")
        sys.exit(1)
    
    try:
        textspotting(image_path, model_dir, heat_map, bbs, output_path, words_output)
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()