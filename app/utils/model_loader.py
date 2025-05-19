"""
Model loader utility for the Smart Trash Classification App
"""

import tensorflow as tf
from config.settings import MODEL_PATH, IMAGE_SIZE


class ModelLoader:
    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the trained model from the specified path"""
        try:
            self.model = tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")

    def predict(self, image):
        """
        Make prediction on the input image

        Args:
            image: Preprocessed image array

        Returns:
            tuple: (predicted_class, confidence_score)
        """
        if self.model is None:
            raise Exception("Model not loaded")

        predictions = self.model.predict(image)
        predicted_class = tf.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])

        return int(predicted_class), confidence
