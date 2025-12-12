"""
Calculate approximate character-level bounding boxes from PaddleOCR response.

The OCR model outputs positions in a normalized feature space (e.g., 157 positions).
This script converts those positions back to pixel coordinates.

Data structures:
    - char_pos_data: Represents a single character's bounding box with position info
    - word_pos_data: Intermediate structure for processing word groups during parsing
"""

from dataclasses import dataclass, field
from typing import List
import numpy as np


@dataclass
class char_pos_data:
    """
    Character bounding box with position information.
    
    Attributes:
        char (str): The character
        bbox (List[int]): Bounding box as [x1, y1, x2, y2]
        center_x (int): X-coordinate of character center
        position_idx (float): Original position index in feature space
    """
    char: str
    bbox: List[int]  # [x1, y1, x2, y2]
    center_x: int
    position_idx: float
    
    def to_dict(self):
        """Convert to dictionary for backward compatibility."""
        return {
            'char': self.char,
            'bbox': self.bbox,
            'center_x': self.center_x,
            'position_idx': self.position_idx
        }


@dataclass
class word_pos_data:
    """
    Intermediate word data structure for processing.
    
    Attributes:
        start_x (float): Word starting x-coordinate in pixels
        end_x (float): Word ending x-coordinate in pixels
        chars (List[str]): List of characters in the word
        positions (List[float]): List of position indices for each character
    """
    start_x: float = 0.0
    end_x: float = 0.0
    chars: List[str] = field(default_factory=list)
    positions: List[float] = field(default_factory=list)
    
    def to_dict(self):
        """Convert to dictionary for backward compatibility."""
        return {
            'start_x': self.start_x,
            'end_x': self.end_x,
            'chars': self.chars,
            'positions': self.positions
        }


def get_char_bboxes(ocr_response, image_width, image_height, return_objects=False):
    """
    Calculate approximate bounding boxes for each character.
    
    Args:
        ocr_response: Tuple from PaddleOCR containing:
            - text (str): Recognized text
            - total_positions (float): Total feature positions (e.g., 157.0)
            - char_groups (list): Grouped characters
            - position_indices (list): Position indices for each group
            - char_types (list): Character type labels
        image_width (int): Original image width in pixels
        image_height (int): Original image height in pixels
        return_objects (bool): If True, return CharacterBBox objects; if False, return dicts
    
    Returns:
        list: Character bounding boxes (CharacterBBox or dict format)
    """
    try:
        text, (total_positions, char_groups, position_indices, char_types) = ocr_response
    except ValueError:
        print("Invalid OCR response format.")
        return []

    # Calculate the ratio to convert position index to pixel coordinate
    ratio = image_width / total_positions

    char_bboxes = []

    for group, positions in zip(char_groups, position_indices):
        # For each character in the group
        for i, char in enumerate(group):
            if i < len(positions):
                # Get position index for this character
                pos_idx = positions[i]
                
                # Calculate x coordinate (center of character)
                x_center = pos_idx * ratio
                
                # Estimate character width
                # If not last char in group, use distance to next char
                # Otherwise, use average character width
                if i + 1 < len(positions):
                    next_pos = positions[i + 1]
                    char_width = (next_pos - pos_idx) * ratio
                else:
                    # Last char: estimate from average or use fixed width
                    if len(positions) > 1:
                        avg_spacing = (positions[-1] - positions[0]) / (len(positions) - 1)
                        char_width = avg_spacing * ratio
                    else:
                        # Single character: use default width
                        char_width = ratio * 4  # Approximate 4 feature units
                
                # Calculate bounding box
                x1 = max(0, x_center - char_width / 2)
                x2 = min(image_width, x_center + char_width / 2)
                y1 = 0  # Top of image (can be adjusted if you have vertical info)
                y2 = image_height  # Bottom of image
                
                char_bbox = char_pos_data(
                    char=char,
                    bbox=[int(x1), int(y1), int(x2), int(y2)],
                    center_x=int(x_center),
                    position_idx=pos_idx
                )
                
                if return_objects:
                    char_bboxes.append(char_bbox)
                else:
                    char_bboxes.append(char_bbox.to_dict())
    
    return char_bboxes


def get_word_bboxes(ocr_response, image_width, image_height):
    """
    Calculate bounding boxes for word groups (more accurate than individual chars).
    
    Args:
        ocr_response: Tuple from PaddleOCR
        image_width (int): Original image width in pixels
        image_height (int): Original image height in pixels
    
    Returns:
        list: Word bounding boxes with text and bbox
    """
    try:
        text, (total_positions, char_groups, position_indices, char_types) = ocr_response
    except ValueError:
        print("Invalid OCR response format.")
        return []
    
    ratio = image_width / total_positions
    
    word_bboxes = []
    
    for group, positions in zip(char_groups, position_indices):
        if not positions:
            continue
        
        # Get word text
        word_text = ''.join(group)
        
        # Calculate bbox from first and last position
        first_pos = positions[0]
        last_pos = positions[-1]
        
        # Estimate character width for padding
        if len(positions) > 1:
            avg_char_width = (last_pos - first_pos) / (len(positions) - 1) * ratio
        else:
            avg_char_width = ratio * 4
        
        # Add padding on both sides
        x1 = max(0, (first_pos * ratio) - avg_char_width / 2)
        x2 = min(image_width, (last_pos * ratio) + avg_char_width / 2)
        y1 = 0
        y2 = image_height
        
        word_bboxes.append({
            'text': word_text,
            'bbox': [int(x1), int(y1), int(x2), int(y2)],
            'char_count': len(group),
            'positions': positions
        })
    
    return word_bboxes

def get_let(word: word_pos_data, image_height: int, image_width: int, ratio: float) -> List[char_pos_data]:
    """
    Calculate character-level bounding boxes for a single word.
    
    Args:
        word: WordData object containing chars and positions
        image_height: Image height in pixels
        image_width: Image width in pixels
        ratio: Conversion ratio (image_width / total_positions)
    
    Returns:
        List of CharacterBBox objects
    """
    chars = word.chars
    positions = word.positions
    
    if len(chars) != len(positions):
        # Handle mismatch by truncating to shorter length
        min_len = min(len(chars), len(positions))
        chars = chars[:min_len]
        positions = positions[:min_len]
    
    char_bboxes = []
    
    for i, (char, pos_idx) in enumerate(zip(chars, positions)):
        # Calculate x coordinate (center of character)
        x_center = pos_idx * ratio
        
        # Estimate character width
        if i + 1 < len(positions):
            # Use distance to next character
            char_width = (positions[i + 1] - pos_idx) * ratio
        elif len(positions) > 1:
            # Last char: use average spacing
            avg_spacing = (positions[-1] - positions[0]) / (len(positions) - 1)
            char_width = avg_spacing * ratio
        else:
            # Single character: use default width (4 feature units)
            char_width = ratio * 4
        
        # Calculate bounding box with bounds checking
        x1 = max(0, x_center - char_width / 2)
        x2 = min(image_width, x_center + char_width / 2)
        y1 = 0
        y2 = image_height
        
        char_bboxes.append(char_pos_data(
            char=char,
            bbox=[int(x1), int(y1), int(x2), int(y2)],
            center_x=int(x_center),
            position_idx=pos_idx
        ))
    
    return char_bboxes


def _separate_spaces_from_groups(data_stream):
    """
    Separate spaces from character groups where they're mixed with other symbols.
    
    Args:
        data_stream (list): List of tuples (chars, positions, type)
    
    Returns:
        list: New data stream with spaces properly separated
    """
    while True:
        new_data_stream = []
        has_mixed_spaces = False
        
        for chars, positions, char_type in data_stream:
            # Check if this group has spaces mixed with other characters
            if len(chars) > 1 and ' ' in chars:
                has_mixed_spaces = True
                # Find all space positions
                space_indices = [i for i, c in enumerate(chars) if c == ' ']
                
                # Split at each space
                last_idx = 0
                for space_idx in space_indices:
                    # Add non-space characters before this space
                    if space_idx > last_idx:
                        new_data_stream.append((
                            chars[last_idx:space_idx],
                            positions[last_idx:space_idx],
                            char_type
                        ))
                    # Add the space as separate group
                    new_data_stream.append(([' '], [positions[space_idx]], 'symbol'))
                    last_idx = space_idx + 1
                
                # Add remaining characters after last space
                if last_idx < len(chars):
                    new_data_stream.append((
                        chars[last_idx:],
                        positions[last_idx:],
                        char_type
                    ))
            else:
                # No mixed spaces, keep as is
                new_data_stream.append((chars, positions, char_type))
        
        # If no mixed spaces found, we're done
        if not has_mixed_spaces:
            break
        
        data_stream = new_data_stream
    
    return data_stream


def _merge_words_without_spaces(data_stream):
    """
    Join consecutive character groups that don't have spaces between them.
    
    Args:
        data_stream (list): List of tuples (chars, positions, type)
    
    Returns:
        list: Data stream with merged word groups
    """
    # Mark groups to be merged
    for i in range(len(data_stream) - 1):
        chars, positions, char_type = data_stream[i]
        
        # Skip if already merged or is a space
        if chars is None or (len(chars) == 1 and chars[0] == ' '):
            continue
        
        # Merge with all following non-space groups
        j = i + 1
        while j < len(data_stream):
            next_chars, next_positions, next_type = data_stream[j]
            
            # Stop at space
            if next_chars and len(next_chars) == 1 and next_chars[0] == ' ':
                break
            
            # Merge this group
            if next_chars is not None:
                chars.extend(next_chars)
                positions.extend(next_positions)
                # Mark as merged
                data_stream[j] = (None, None, None)
            
            j += 1
    
    return data_stream


def _extract_words_from_stream(data_stream, ratio, image_width, image_height):
    """
    Extract word structures from the processed data stream.
    
    Args:
        data_stream (list): Processed data stream with merged groups
        ratio (float): Position to pixel conversion ratio
        image_width (int): Image width in pixels
        image_height (int): Image height in pixels
    
    Returns:
        list: Word dictionaries with bbox and letters
    """
    words = [word_pos_data()]
    
    for chars, positions, char_type in data_stream:
        # Skip merged groups
        if chars is None or positions is None:
            continue
        
        # Space encountered - finalize current word and start new one
        if len(chars) == 1 and chars[0] == ' ':
            if positions:
                words[-1].end_x = positions[0] * ratio
                new_word = word_pos_data(start_x=positions[0] * ratio + 1)
                words.append(new_word)
            continue
        
        # Add characters to current word
        words[-1].chars.extend(chars)
        words[-1].positions.extend(positions)
    
    # Finalize last word
    if words and words[-1].end_x == 0:
        words[-1].end_x = image_width
    
    # Convert to final format with bboxes
    result_words = []
    for word in words:
        if len(word.chars) > 0:
            result_words.append({
                "bbox": [
                    int(word.start_x),
                    0,
                    int(word.end_x),
                    image_height
                ],
                "letters": get_let(word, image_height, image_width, ratio)
            })
    
    return result_words


def process_ocr_response(ocr_response, image_width, image_height):
    """
    Process OCR response to extract word-level bounding boxes with character details.
    
    This function:
    1. Separates spaces from mixed character groups
    2. Merges consecutive character groups without spaces into words
    3. Calculates bounding boxes for words and individual characters
    
    Args:
        ocr_response: Tuple from PaddleOCR containing:
            - text (str): Recognized text
            - (total_positions, char_groups, position_indices, char_types)
        image_width (int): Image width in pixels
        image_height (int): Image height in pixels
    
    Returns:
        list: Words with structure:
            - 'bbox': [x1, y1, x2, y2] word bounding box
            - 'letters': List of character bboxes from get_let()
    """
    try:
        text, (total_positions, char_groups, position_indices, char_types) = ocr_response
    except ValueError:
        print("Invalid OCR response format.")
        return []

    # Calculate position to pixel conversion ratio
    ratio = image_width / total_positions

    # Create initial data stream
    data_stream = list(zip(char_groups, position_indices, char_types))

    # Step 1: Separate spaces from mixed groups
    data_stream = _separate_spaces_from_groups(data_stream)

    # Step 2: Merge consecutive non-space groups into words
    data_stream = _merge_words_without_spaces(data_stream)

    # Step 3: Extract word structures with bboxes
    words = _extract_words_from_stream(data_stream, ratio, image_width, image_height)

    return words


def visualize_bboxes(image_path, char_bboxes, output_path=None):
    """
    Draw bounding boxes on the image for visualization.
    
    Args:
        image_path (str): Path to input image
        char_bboxes (list): Character bboxes from get_char_bboxes()
        output_path (str): Optional path to save visualization
    
    Returns:
        numpy.ndarray: Image with drawn bboxes
    """
    import cv2
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    # Draw each character bbox
    for i, item in enumerate(char_bboxes):
        bbox = item['bbox']
        #char = item['char']

        color = colors[i%len(colors)]
        # Draw rectangle
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 1)
        
        # Put character label
        #cv2.putText(img, char, (bbox[0], bbox[1] - 2),
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    
    if output_path:
        cv2.imwrite(output_path, img)
        print(f"Saved visualization to: {output_path}")
    
    return img

def visualize_words(image_path, words, output_path=None):
    import cv2

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    w_color = (0, 255, 255)
    let_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    # Draw each word bbox
    for i, word in enumerate(words):
        bbox = word['bbox']
        x1, y1, x2, y2 = bbox
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), w_color, 2)
        # Draw letters
        for j, letter in enumerate(word['letters']):
            bbox = letter.bbox
            color = let_colors[j % len(let_colors)]
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 1)

    if output_path:
        cv2.imwrite(output_path, img)
        print(f"Saved visualization to: {output_path}")

    return img


# Example usage
if __name__ == '__main__':
    # Example response from your OCR
    ocr_response = (
        'Heparin 1,000 Units 2 Units/Ml Inj (00409-1005-01',
        [157.0,
        [['H', 'e', 'p', 'a', 'r', 'i', 'n'],
         [' '], ['1'], [','],
         ['0', '0', '0'],
         [' '],
         ['U', 'n', 'i', 't', 's'],
         [' '], ['2'], [' '],
         ['U', 'n', 'i', 't', 's'],
         ['/'], ['M', 'l'], [' '],
         ['I', 'n', 'j'], [' ', '('],
         ['0', '0', '4', '0', '9', '-', '1', '0', '0', '5', '-', '0', '1']],
        [[2, 6, 10, 14, 17, 19, 22], [25], [28], [31], [33, 37, 41], [44], 
         [47, 52, 55, 57, 59], [62], [65], [68], [71, 75, 78, 80, 83], [86], 
         [90, 95], [96], [99, 101, 104], [106, 108], 
         [111, 115, 119, 122, 126, 129, 133, 137, 140, 144, 147, 151, 155]],
        ['en&num', 'symbol', 'en&num', 'symbol', 'en&num', 'symbol', 'en&num', 
         'symbol', 'en&num', 'symbol', 'en&num', 'symbol', 'en&num', 'symbol', 
         'en&num', 'symbol', 'en&num']]
    )
    
    # Assume image dimensions (replace with actual values)
    image_width = 341
    image_height = 13
    
    # Get character-level bboxes
    char_bboxes = get_char_bboxes(ocr_response, image_width, image_height)
    
    print("Character-level bounding boxes:")
    print("=" * 60)
    for item in char_bboxes[:10]:  # Show first 10
        print(f"Char: '{item['char']}' -> bbox: {item['bbox']}, center_x: {item['center_x']}")
    print(f"... (total {len(char_bboxes)} characters)")
    
    print("\n" + "=" * 60)
    
    # Get word-level bboxes (often more useful)
    word_bboxes = get_word_bboxes(ocr_response, image_width, image_height)

    words = process_ocr_response(ocr_response, image_width, image_height)
    visualize_words("d:/projects/issues/jira-515/invest/advent.cmmmmm_UB_IB_9_30_24.pdf-000042.png_886791.png",
                     words,
                     "d:/projects/issues/jira-515/invest/advent.cmmmmm_UB_IB_9_30_24.pdf-000042.png_886791.words.png")
    
    print("\nWord-level bounding boxes:")
    print("=" * 60)
    for item in word_bboxes:
        print(f"Word: '{item['text']:15s}' -> bbox: {item['bbox']}")
    
    print("\n" + "=" * 60)
    print("\nTo use with your image:")
    print("  char_bboxes = get_char_bboxes(ocr_response, img.shape[1], img.shape[0])")
    print("  word_bboxes = get_word_bboxes(ocr_response, img.shape[1], img.shape[0])")
    print("\nTo visualize:")
    print("  visualize_bboxes('input.png', char_bboxes, 'output.png')")
    visualize_bboxes("d:/projects/issues/jira-515/invest/advent.cmmmmm_UB_IB_9_30_24.pdf-000042.png_886791.png",
                     char_bboxes,
                     "d:/projects/issues/jira-515/invest/advent.cmmmmm_UB_IB_9_30_24.pdf-000042.png_886791.bbox_viz.png")
    visualize_bboxes("d:/projects/issues/jira-515/invest/advent.cmmmmm_UB_IB_9_30_24.pdf-000042.png_886791.png",
                     word_bboxes,
                     "d:/projects/issues/jira-515/invest/advent.cmmmmm_UB_IB_9_30_24.pdf-000042.png_886791.bbox_viz_words.png")
