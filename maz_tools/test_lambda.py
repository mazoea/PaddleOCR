"""
Test script for Lambda function locally
"""
import base64
import json
import os
from lambda_handler import lambda_handler


def test_local_image(image_path):
    """Test Lambda handler with a local image"""
    print(f"Testing with image: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    # Read and encode image
    with open(image_path, 'rb') as f:
        image_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    # Create test event
    event = {
        'image': image_base64,
        'return_format': 'base64'
    }
    
    # Call Lambda handler
    print("\nInvoking Lambda handler...")
    response = lambda_handler(event, None)
    
    print(f"\nResponse status: {response['statusCode']}")
    
    # Parse and save result
    if response['statusCode'] == 200:
        body = json.loads(response['body'])
        
        if body['success']:
            result = body['result']
            
            # Decode and save result image
            result_image = base64.b64decode(result['image'])
            output_path = image_path.replace('.', '_lambda_result.')
            
            with open(output_path, 'wb') as f:
                f.write(result_image)
            
            print(f"\n✅ Success!")
            print(f"   Output saved to: {output_path}")
            print(f"   Detected regions: {result['num_regions']}")
            print(f"   Detection time: {result['detection_time']:.3f} seconds")
            print(f"   Image size: {result['image_size']['width']}x{result['image_size']['height']}")
        else:
            print(f"\n❌ Error: {body.get('error', 'Unknown error')}")
    else:
        body = json.loads(response['body'])
        print(f"\n❌ Error: {body.get('error', 'Unknown error')}")


def test_multiple_images(directory, pattern="*.png"):
    """Test Lambda handler with multiple images"""
    import glob
    
    images = glob.glob(os.path.join(directory, pattern))
    
    if not images:
        print(f"No images found in {directory} matching {pattern}")
        return
    
    print(f"Found {len(images)} images to test")
    
    for i, image_path in enumerate(images[:5], 1):  # Test first 5 images
        print(f"\n{'='*60}")
        print(f"Test {i}/{min(len(images), 5)}")
        test_local_image(image_path)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python test_lambda.py <image_path>")
        print("  python test_lambda.py --batch <directory>")
        print("\nExamples:")
        print("  python test_lambda.py advent.44-000001.png")
        print("  python test_lambda.py --batch qa_input")
        sys.exit(1)
    
    if sys.argv[1] == "--batch":
        directory = sys.argv[2] if len(sys.argv) > 2 else "qa_input"
        test_multiple_images(directory)
    else:
        test_local_image(sys.argv[1])
