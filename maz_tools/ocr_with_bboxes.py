"""
OCR with Bounding Boxes Extraction

This script demonstrates how to extract both text and bounding boxes from images
using PaddleOCR's full pipeline (detection + recognition).

Unlike TextRecognition which only recognizes text from pre-cropped images,
PaddleOCR pipeline detects text regions first and then recognizes them.

Key findings:
1. TextRecognition: Only for recognition of pre-cropped word images
2. PaddleOCR: Full pipeline with detection + recognition
3. return_word_box: Can provide word-level boxes within each detected region
"""

from paddleocr import PaddleOCR
import json
from pathlib import Path


def ocr_with_bboxes(image_path, model_dir=None, device='cpu', use_gpu=False):
    """
    Perform OCR on an image and extract both text and bounding boxes.
    
    Args:
        image_path: Path to the image file
        model_dir: Optional custom model directory
        device: 'cpu' or 'gpu'
        use_gpu: Deprecated parameter, use device='gpu' instead
        
    Returns:
        dict: Contains 'dt_polys' (bounding boxes), 'rec_texts' (recognized texts), 
              and 'rec_scores' (confidence scores)
    """
    # Initialize PaddleOCR with full pipeline
    # This includes both detection (to find text regions) and recognition (to read them)
    ocr = PaddleOCR(
        lang='en',  # Change to 'ch' for Chinese, or other supported languages
        ocr_version='PP-OCRv5',
        device=device,
        # Optional: specify custom models
        # text_detection_model_dir=model_dir,
        # text_recognition_model_dir=model_dir,
    )
    
    # Run OCR - by default returns detection boxes and recognized text
    result = ocr.predict(image_path)
    
    return result[0] if result else None


def ocr_with_word_boxes(image_path, model_dir=None, device='cpu'):
    """
    Perform OCR with word-level bounding boxes.
    
    The return_word_box parameter provides additional word-level segmentation
    within each detected text region. This is useful when a detected region
    contains multiple words.
    
    Args:
        image_path: Path to the image file
        model_dir: Optional custom model directory
        device: 'cpu' or 'gpu'
        
    Returns:
        dict: Contains 'dt_polys', 'rec_texts', 'rec_scores', and potentially
              word-level information
    """
    ocr = PaddleOCR(
        lang='en',
        ocr_version='PP-OCRv5',
        device=device,
    )
    
    # Use return_word_box=True to get word-level boxes
    result = ocr.predict(image_path, return_word_box=True)
    
    return result[0] if result else None


def format_ocr_result(result):
    """
    Format OCR result into a readable structure.
    
    Args:
        result: OCR result dictionary
        
    Returns:
        list: List of detected text items with boxes and text
    """
    if not result:
        return []
    
    formatted = []
    dt_polys = result.get('dt_polys', [])
    rec_texts = result.get('rec_texts', [])
    rec_scores = result.get('rec_scores', [])
    
    for i, (poly, text, score) in enumerate(zip(dt_polys, rec_texts, rec_scores)):
        formatted.append({
            'index': i,
            'text': text,
            'confidence': float(score),
            'bbox': poly.tolist() if hasattr(poly, 'tolist') else poly,
        })
    
    return formatted


def main():
    """Example usage"""
    # Example 1: Basic OCR with bounding boxes
    image_path = r"d:\projects\PaddleOCR\maz_tools\test_image.jpg"
    
    print("=" * 80)
    print("Example 1: Basic OCR with Detection + Recognition")
    print("=" * 80)
    
    result = ocr_with_bboxes(image_path, device='cpu')
    
    if result:
        print(f"\nFound {len(result.get('rec_texts', []))} text regions\n")
        
        formatted = format_ocr_result(result)
        for item in formatted:
            print(f"Text {item['index']}: '{item['text']}'")
            print(f"  Confidence: {item['confidence']:.3f}")
            print(f"  Bounding box: {item['bbox']}")
            print()
        
        # Save to JSON
        output_path = Path(image_path).with_suffix('.ocr_result.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(formatted, f, indent=2, ensure_ascii=False)
        print(f"Results saved to: {output_path}")
    else:
        print("No text detected in image")
    
    print("\n" + "=" * 80)
    print("Example 2: OCR with Word-Level Boxes")
    print("=" * 80)
    
    result_with_words = ocr_with_word_boxes(image_path, device='cpu')
    
    if result_with_words:
        print(f"\nFound {len(result_with_words.get('rec_texts', []))} text regions")
        print("(with word-level segmentation)")
        
        # The result structure with return_word_box=True may contain additional
        # word-level information in the recognition results
        print("\nNote: Word-level boxes provide finer granularity within each")
        print("detected text region, useful for multi-word lines.")


if __name__ == '__main__':
    main()


"""
SUMMARY:

1. TextRecognition vs PaddleOCR:
   - TextRecognition: Only recognizes text from pre-cropped images
   - PaddleOCR: Full pipeline = Detection + Recognition
   
2. To get bounding boxes with text:
   - Use PaddleOCR (not TextRecognition)
   - Output contains 'dt_polys' (bounding boxes) + 'rec_texts' + 'rec_scores'
   
3. For word-level boxes:
   - Use return_word_box=True parameter
   - Provides finer segmentation within detected regions
   
4. Output format:
   - dt_polys: List of polygon coordinates (usually 4 points per box)
   - rec_texts: List of recognized text strings
   - rec_scores: List of confidence scores
   
5. Usage in your case:
   Replace TextRecognition with PaddleOCR in simple_ocr.py:
   
   Before:
       from paddleocr import TextRecognition
       model = TextRecognition(model_name=model_name, device='cpu')
       result = model.predict(input=img_path, batch_size=1)
       # Only gets: rec_text, rec_score
   
   After:
       from paddleocr import PaddleOCR
       ocr = PaddleOCR(lang='en', device='cpu')
       result = ocr.predict(img_path)
       # Gets: dt_polys (bboxes), rec_texts, rec_scores
"""
