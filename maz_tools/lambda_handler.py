"""
AWS Lambda handler for text detection using PaddleOCR
This handler processes images and returns them with detected text bounding boxes.
"""
import json
import base64
import os
import tempfile
import cv2
import numpy as np
import time

# CRITICAL: Set environment variables BEFORE importing PaddleOCR
# Lambda only allows writing to /tmp, so redirect all cache/temp directories there
os.environ['HOME'] = '/tmp'
os.environ['TMPDIR'] = '/tmp'
os.environ['TEMP'] = '/tmp'
os.environ['TMP'] = '/tmp'
os.environ['PADDLEX_HOME'] = '/tmp/paddlex'
os.environ['PADDLEHUB_HOME'] = '/tmp/paddlehub'
os.environ['PADDLE_HOME'] = '/tmp/paddle'

# PaddlePaddle CPU optimization flags for Lambda
# These prevent segmentation faults on Lambda's CPU architecture
os.environ['FLAGS_use_mkldnn'] = 'false'  # Disable MKL-DNN to avoid CPU compatibility issues
os.environ['FLAGS_cpu_deterministic'] = 'true'  # Force deterministic behavior
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Allow duplicate library loading
os.environ['OMP_NUM_THREADS'] = '1'  # Limit OpenMP threads
os.environ['MKL_NUM_THREADS'] = '1'  # Limit MKL threads
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # Limit OpenBLAS threads

# Create necessary directories in /tmp
os.makedirs('/tmp/paddlex', exist_ok=True)
os.makedirs('/tmp/paddlehub', exist_ok=True)
os.makedirs('/tmp/paddle', exist_ok=True)

from paddleocr import TextDetection


# Initialize the detector globally for Lambda container reuse
print("Initializing TextDetection model...")
DETECTOR = None


def get_detector():
    """Get or initialize the detector (singleton pattern for Lambda reuse)"""
    global DETECTOR
    if DETECTOR is None:
        print("Initializing TextDetection with Lambda-compatible settings...")
        model_dir = os.environ.get('MODEL_DIR', './PP-OCRv5_mobile_det')
        
        try:
            DETECTOR = TextDetection(
                device='cpu',
                model_dir=model_dir,
                model_name="PP-OCRv5_mobile_det",
                limit_side_len=64,
                limit_type='min',
                thresh=0.3,
                box_thresh=0.6,
                unclip_ratio=1.5,
                cpu_threads=1,
                enable_mkldnn=False
            )
            print("TextDetection model initialized successfully")
        except Exception as e:
            print(f"Error initializing TextDetection: {e}")
            import traceback
            traceback.print_exc()
            raise
    return DETECTOR


def detect_text_regions(image_data, return_format='base64'):
    """
    Detect text regions in an image and draw bounding boxes.
    
    Args:
        image_data: Image data (numpy array or file path)
        return_format: 'base64' or 'bytes'
    
    Returns:
        dict: Result containing processed image and metadata
    """
    s_handling = time.time()
    detector = get_detector()
    
    # Read image if it's a path
    if isinstance(image_data, str):
        img = cv2.imread(image_data)
    else:
        img = image_data
    
    if img is None:
        raise ValueError("Could not read image data")
    
    # if image is too large, normalize it that one side has max 1920px
    max_side_len = 1280
    h, w, _ = img.shape
    if max(h, w) > max_side_len:
        ratio = max_side_len / max(h, w)
        img = cv2.resize(img, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_AREA)
        #print(f"Resized image to {img.shape[1]}x{img.shape[0]} (width x height)")
    
    print(f"Image size: {img.shape[1]}x{img.shape[0]} (width x height)")
    
    # Perform text detection
    start_time = time.time()
    results = detector.predict(img)
    detection_time = time.time() - start_time
    
    detected_regions = results[0]
    num_regions = len(detected_regions['dt_polys'])
    print(f"Detected {num_regions} text regions in {detection_time:.3f} seconds")
    handling_time = time.time() - s_handling
    print(f"Detection plus model loading time: {handling_time:.3f} seconds")
    
    # Draw polygons around each detected text region
    output_image = img.copy()

    bboxes = []
    for bbox in detected_regions["dt_polys"]:
        # Convert bbox to numpy array
        box = np.reshape(np.array(bbox), [-1, 1, 2]).astype(np.int64)
        # Draw blue polygon (BGR format, so blue is (255, 0, 0))
        output_image = cv2.polylines(output_image, [box], True, (255, 0, 0), 2)
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
    
    # Encode the output image
    _, buffer = cv2.imencode('.png', output_image)
    
    if return_format == 'base64':
        image_encoded = base64.b64encode(buffer).decode('utf-8')
    else:
        image_encoded = buffer.tobytes()
    
    return {
        'image': image_encoded,
        'num_regions': num_regions,
        'detection_time': detection_time,
        "handling_time": handling_time,
        'image_size': {'width': img.shape[1], 'height': img.shape[0]},
        "bboxes": bboxes
    }


def lambda_handler(event, context):
    """
    AWS Lambda handler function.
    
    Expected event format:
    {
        "image": "<base64 encoded image>",
        "return_format": "base64" (optional, default: "base64")
    }
    
    Or for S3 trigger:
    {
        "Records": [{"s3": {"bucket": {"name": "..."}, "object": {"key": "..."}}}]
    }
    """
    try:
        print("Processing Lambda request...")
        
        # Handle direct invocation with base64 image
        if 'image' in event:
            print("Processing direct invocation with base64 image")
            image_base64 = event['image']
            return_format = event.get('return_format', 'base64')
            
            # Decode base64 image
            image_bytes = base64.b64decode(image_base64)
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Process the image
            result = detect_text_regions(img, return_format=return_format)
            
            return {
                'statusCode': 200,
                'headers': {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*'
                },
                'body': json.dumps({
                    'success': True,
                    'result': result
                })
            }
        
        # Handle S3 trigger
        elif 'Records' in event:
            import boto3
            s3 = boto3.client('s3')
            
            results = []
            for record in event['Records']:
                bucket = record['s3']['bucket']['name']
                key = record['s3']['object']['key']
                
                print(f"Processing S3 object: s3://{bucket}/{key}")
                
                # Download image from S3
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                    s3.download_file(bucket, key, tmp_file.name)
                    
                    # Process the image
                    result = detect_text_regions(tmp_file.name, return_format='bytes')
                    
                    # Upload result back to S3
                    output_key = key.replace('.', '_detected.')
                    s3.put_object(
                        Bucket=bucket,
                        Key=output_key,
                        Body=result['image'],
                        ContentType='image/png'
                    )
                    
                    # Clean up temp file
                    os.unlink(tmp_file.name)
                    
                    results.append({
                        'input': f"s3://{bucket}/{key}",
                        'output': f"s3://{bucket}/{output_key}",
                        'num_regions': result['num_regions'],
                        'detection_time': result['detection_time']
                    })
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'success': True,
                    'results': results
                })
            }
        
        else:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'success': False,
                    'error': 'Invalid request format. Expected "image" or "Records".'
                })
            }
    
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'success': False,
                'error': str(e)
            })
        }


# For local testing
if __name__ == "__main__":
    # Test with a local image
    test_image_path = "./qa_input/advent.44-000001.png"
    
    if os.path.exists(test_image_path):
        print(f"Testing with local image: {test_image_path}")
        
        # Read and encode image
        with open(test_image_path, 'rb') as f:
            image_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        # Create test event
        event = {
            'image': image_base64,
            'return_format': 'base64'
        }
        
        # Call handler
        response = lambda_handler(event, None)
        print(f"Response status: {response['statusCode']}")
        
        # Save result image
        if response['statusCode'] == 200:
            body = json.loads(response['body'])
            result_image = base64.b64decode(body['result']['image'])
            
            with open('test_lambda_output.png', 'wb') as f:
                f.write(result_image)
            
            print(f"Result saved to: test_lambda_output.png")
            print(f"Detected regions: {body['result']['num_regions']}")
    else:
        print(f"Test image not found: {test_image_path}")
