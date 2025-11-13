"""
Example client script for invoking the text detection Lambda function
"""
import boto3
import base64
import json
import sys
import os


def invoke_lambda(image_path, function_name='te-advent-tm-v2', region='eu-central-1'):
    """
    Invoke the text detection Lambda function with an image.
    
    Args:
        image_path: Path to the input image
        function_name: Name of the Lambda function
        region: AWS region
    
    Returns:
        Path to the output image
    """
    # Initialize Lambda client
    lambda_client = boto3.client('lambda', region_name=region)
    
    # Read and encode image
    #print(f"Reading image: {image_path}")
    with open(image_path, 'rb') as f:
        image_base64 = base64.b64encode(f.read()).decode('utf-8')
    
    # Create payload
    payload = {
        'image': image_base64,
        'return_format': 'base64'
    }
    
    # Invoke Lambda
    #print(f"Invoking Lambda function: {function_name}")
    response = lambda_client.invoke(
        FunctionName=function_name,
        InvocationType='RequestResponse',
        LogType='Tail',  # Request execution logs
        Payload=json.dumps(payload)
    )
    
    # Extract memory usage from logs
    memory_used = None
    memory_size = None
    if 'LogResult' in response:
        logs = base64.b64decode(response['LogResult']).decode('utf-8')
        # Parse REPORT line for memory info
        for line in logs.split('\n'):
            if line.startswith('REPORT'):
                # Extract memory metrics from REPORT line
                if 'Memory Size:' in line and 'Max Memory Used:' in line:
                    import re
                    memory_size_match = re.search(r'Memory Size: (\d+) MB', line)
                    memory_used_match = re.search(r'Max Memory Used: (\d+) MB', line)
                    if memory_size_match:
                        memory_size = int(memory_size_match.group(1))
                    if memory_used_match:
                        memory_used = int(memory_used_match.group(1))
    
    # Parse response
    result = json.loads(response['Payload'].read())
    try:
        if result['statusCode'] != 200:
            print(result)
            raise Exception(f"Lambda invocation failed: {result}")
    except Exception as e:
        print(result)
        raise Exception(f"Lambda invocation failed: {result}")
    
    body = json.loads(result['body'])
    
    if not body['success']:
        raise Exception(f"Lambda processing failed: {body.get('error', 'Unknown error')}")
    
    # Print results
    #print(f"\n✅ Success!")
    #print(f"   Output saved to: {output_path}")
    #print(f"   Detected regions: {body['result']['num_regions']}")
    #print(f"   Detection time: {body['result']['detection_time']:.3f} seconds")
    #print(f"   Image size: {body['result']['image_size']['width']}x{body['result']['image_size']['height']}")
    
    # Print memory usage if available
    #if memory_used and memory_size:
    #    memory_percent = (memory_used / memory_size) * 100
    #    print(f"\n📊 Memory Usage:")
    #    print(f"   Allocated: {memory_size} MB")
    #    print(f"   Used: {memory_used} MB ({memory_percent:.1f}%)")
    #    if memory_percent > 80:
    #        print(f"   ⚠️  High memory usage! Consider increasing Lambda memory size.")
    #    elif memory_percent < 50:
    #        print(f"   ℹ️  You could reduce Lambda memory size to save costs.")
    
    print(f"{image_path},{body['result']['text']},{body['result']['conf']:.3f},{body['result']['detection_time']:.3f},{body['result']['handling_time']:.3f},{memory_used or '0'},{body['result']['image_size']['width']}x{body['result']['image_size']['height']}")
    
    return


def main():
    """Main function to handle command line arguments"""
    if len(sys.argv) < 2:
        print("Usage: python invoke_lambda.py <image_path> [function_name] [region]")
        print("\nExample:")
        print("  python invoke_lambda.py input.png")
        print("  python invoke_lambda.py input.png te-advent-tm-v2 eu-central-1")
        sys.exit(1)
    
    image_path = sys.argv[1]
    #
    function_name = sys.argv[2] if len(sys.argv) > 2 else 'te-advent-tm-v2'
    region = sys.argv[3] if len(sys.argv) > 3 else 'eu-central-1'
    
    # Check if input file exists
    if not os.path.exists(image_path):
        print(f"Error: Input file '{image_path}' does not exist!")
        sys.exit(1)

    try:
        invoke_lambda(image_path, function_name, region)
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
