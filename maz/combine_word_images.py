"""
CombineWordImages - Augmentation operator for combining single-word images.

This module provides functionality to combine multiple single-word images
into artificial multi-word images during training, all contained within
the maz folder to minimize changes to the original PaddleOCR codebase.
"""

import cv2
import numpy as np
import random


class CombineWordImages(object):
    """
    Combine multiple single-word images into one artificial multi-word image.
    
    This operator randomly selects 2-N word images from ext_data and combines them
    horizontally with spacing between them. The ground truth labels are joined with spaces.
    
    Args:
        prob (float): Probability to apply this augmentation. Default: 0.5
        min_words (int): Minimum number of words to combine. Default: 2
        max_words (int): Maximum number of words to combine. Default: 4
        space_range (tuple): Range for random spacing between words in pixels (min, max). Default: (5, 20)
        target_height (int or None): Target height for all word images. If None, images are scaled
            to match the tallest image in the bundle (keeping aspect ratio). Default: None
        pad_color (int): Background color for padding (0-255 for grayscale/BGR). Default: 255 (white)
    """

    def __init__(
        self,
        prob=0.5,
        min_words=2,
        max_words=4,
        space_range=(5, 20),
        target_height=None,
        pad_color=255,
        **kwargs
    ):
        self.prob = prob
        self.min_words = max(2, min_words)  # At least 2 words
        self.max_words = max(self.min_words, max_words)
        self.space_range = space_range
        self.target_height = target_height
        self.pad_color = pad_color

    def _resize_keep_aspect(self, img, target_height):
        """Resize image to target height while keeping aspect ratio."""
        h, w = img.shape[:2]
        if h == 0:
            return img
        ratio = target_height / h
        new_w = int(w * ratio)
        if new_w == 0:
            new_w = 1
        resized = cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_LINEAR)
        return resized

    def _decode_image_bytes(self, img_bytes):
        """Decode image bytes to numpy array."""
        if isinstance(img_bytes, bytes):
            img = np.frombuffer(img_bytes, dtype="uint8")
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            return img
        elif isinstance(img_bytes, np.ndarray):
            return img_bytes
        return None

    def _combine_images_horizontal(self, images, spacings):
        """
        Combine multiple images horizontally with specified spacings.
        
        Args:
            images: List of numpy arrays (images)
            spacings: List of spacing values between images
            
        Returns:
            Combined image as numpy array
        """
        if len(images) == 0:
            return None

        # Determine target height
        if self.target_height is None:
            # Use the height of the tallest image in the bundle
            target_height = max(img.shape[0] for img in images if img is not None)
        else:
            target_height = self.target_height

        # Resize all images to target height
        resized_images = []
        for img in images:
            if img is None:
                continue
            resized = self._resize_keep_aspect(img, target_height)
            resized_images.append(resized)

        if len(resized_images) == 0:
            return None

        # Normalize all images to 3 channels (BGR) for consistency
        # This handles mixed grayscale and color images
        normalized_images = []
        for img in resized_images:
            if len(img.shape) == 2:
                # Grayscale to BGR
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.shape[2] == 1:
                # Single channel to BGR
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            normalized_images.append(img)

        # Calculate total width
        total_width = sum(img.shape[1] for img in normalized_images)
        total_width += sum(spacings)

        # Create blank canvas (always 3 channels for consistency)
        combined = np.ones((target_height, total_width, 3), dtype=np.uint8) * self.pad_color

        # Place images on canvas
        x_offset = 0
        for i, img in enumerate(normalized_images):
            w = img.shape[1]
            combined[:, x_offset:x_offset + w] = img
            x_offset += w
            if i < len(spacings):
                x_offset += spacings[i]

        return combined

    def __call__(self, data):
        """
        Apply the augmentation.
        
        Args:
            data: Dict containing:
                - 'image': bytes or numpy array of the main image
                - 'label': str, the ground truth text
                - 'ext_data': list of dicts with 'image' and 'label' for additional samples
                
        Returns:
            Modified data dict with combined image and label, or original data if not applied
        """
        # Handle case where data might be a list (from previous transforms)
        if not isinstance(data, dict):
            return data
        
        # ALWAYS decode ext_data images first (for downstream operators like RecConAug)
        ext_data = data.get("ext_data", [])
        for ext_item in ext_data:
            if isinstance(ext_item.get("image"), bytes):
                decoded = self._decode_image_bytes(ext_item["image"])
                if decoded is not None:
                    ext_item["image"] = decoded
        
        # Check if we should apply the combining augmentation
        if random.random() > self.prob:
            return data

        # Check if ext_data has enough samples to combine
        if len(ext_data) < self.min_words - 1:
            # Not enough samples to combine
            return data

        # Decide how many words to combine
        available_words = min(len(ext_data) + 1, self.max_words)  # +1 for the main image
        n_words = random.randint(self.min_words, available_words)

        # Select random samples from ext_data (n_words - 1, because we include the main image)
        selected_indices = random.sample(range(len(ext_data)), n_words - 1)

        # Collect images and labels
        images = []
        labels = []

        # Get main image (should already be decoded by DecodeImage operator)
        main_img = self._decode_image_bytes(data["image"])
        if main_img is None:
            return data
        images.append(main_img)
        labels.append(data["label"])

        # Collect selected ext_data images (already decoded above)
        for idx in selected_indices:
            ext_img = ext_data[idx]["image"]
            # ext_img should already be numpy array at this point
            if ext_img is not None and isinstance(ext_img, np.ndarray):
                images.append(ext_img)
                labels.append(ext_data[idx]["label"])

        # If we couldn't decode enough images, return original
        if len(images) < 2:
            return data

        # Generate random spacings between words
        spacings = [random.randint(self.space_range[0], self.space_range[1]) 
                    for _ in range(len(images) - 1)]

        # Combine images
        combined_img = self._combine_images_horizontal(images, spacings)
        # cv2.imwrite("d:/tmp/debug.png", combined_img)

        if combined_img is None:
            return data

        # Combine labels with spaces
        combined_label = " ".join(labels)

        # Update data with combined image (as numpy array, not bytes)
        data["image"] = combined_img
        data["label"] = combined_label

        return data
