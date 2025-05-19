"""
Image processing utility for the Smart Trash Classification App
"""

import cv2
import numpy as np
from PIL import Image
from config.settings import IMAGE_SIZE


class ImageProcessor:
    @staticmethod
    def preprocess_image(image):
        """
        Preprocess the uploaded image for model prediction

        Args:
            image: Uploaded image file

        Returns:
            numpy.ndarray: Preprocessed image array
        """
        # Convert to PIL Image
        if isinstance(image, bytes):
            image = Image.open(image)

        # Convert to numpy array
        image_array = np.array(image)

        # Convert to RGB if grayscale
        if len(image_array.shape) == 2:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        elif image_array.shape[2] == 4:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)

        # Resize image
        image_array = cv2.resize(image_array, IMAGE_SIZE)

        # Normalize pixel values
        image_array = image_array.astype(np.float32) / 255.0

        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        return image_array
