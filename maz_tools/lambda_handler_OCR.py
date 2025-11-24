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

from paddleocr import TextRecognition

# Initialize the detector globally for Lambda container reuse
print("Initializing TextDetection model...")
TEXT_REC = None


def get_convertor():
    """Get or initialize the detector (singleton pattern for Lambda reuse)"""
    global TEXT_REC
    if TEXT_REC is None:
        print("Initializing TextDetection with Lambda-compatible settings...")
        model_dir = os.environ.get('MODEL_DIR', './___PP-OCRv5_mobile_rec_infer')
        if "PP-OCRv5_mobile_rec" in model_dir:
            model_name = "PP-OCRv5_mobile_rec"
        if "PP-OCRv5_server_rec" in model_dir:
            model_name = "PP-OCRv5_server_rec"
        if "en_PP-OCRv5_mobile_rec" in model_dir:
            model_name = "en_PP-OCRv5_mobile_rec"
        if "latin_PP-OCRv5_mobile_rec" in model_dir:
            model_name = "latin_PP-OCRv5_mobile_rec"

        try:
            TEXT_REC = TextRecognition(
                device='cpu',
                model_dir=model_dir,
                model_name=model_name,
                cpu_threads=1,
                mkldnn_cache_capacity=0,
                enable_mkldnn=False
            )
            print("TextRecognition model initialized successfully")
        except Exception as e:
            print(f"Error initializing TextRecognition: {e}")
            import traceback
            traceback.print_exc()
            raise
    return TEXT_REC


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
    convertor = get_convertor()
    
    # Read image if it's a path
    if isinstance(image_data, str):
        img = cv2.imread(image_data)
    else:
        img = image_data
    
    if img is None:
        raise ValueError("Could not read image data")
    
    # Perform text detection
    start_time = time.time()
    results = convertor.predict(input=img, batch_size=1)
    detection_time = time.time() - start_time
    
    word_info = results[0]
    text, conf = (word_info.get('rec_text', 'no_text'), word_info.get('rec_score', 0.0))
    handling_time = time.time() - s_handling
    print(f"Detection plus model loading time: {handling_time:.3f} seconds")
    
    return {
        'text': text,
        'conf': conf,
        'detection_time': detection_time,
        "handling_time": handling_time,
        'image_size': {'width': img.shape[1], 'height': img.shape[0]}
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
            print(body)
    else:
        print(f"Test image not found: {test_image_path}")
