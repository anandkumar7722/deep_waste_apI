# coding: utf-8
"""Module that does all the ML trained model prediction heavy lifting."""
from logging import Logger, getLogger
from datetime import datetime, date
from os.path import normpath, join, dirname
from typing import Any, Dict

import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

log: Logger = getLogger(__name__)


def full_path(filename: str) -> str:
    """Returns the full normalised path of a file in the same folder as this module."""
    return normpath(join(dirname(__file__), filename))


MODEL: Any = None


def init() -> None:
    """Loads the ML trained model (plus ancillary files) from file."""
    global MODEL

    model_path = full_path("garbage_model")
    log.debug("Initialize model from file %s", model_path)

    if not os.path.exists(model_path):
        log.error(f"Model folder not found: {model_path}")
        raise FileNotFoundError(f"Model folder not found: {model_path}")

    try:
        # Try loading as a SavedModel
        MODEL = tf.saved_model.load(model_path)
        log.info("Model loaded successfully (tf.saved_model.load).")
    except Exception as e:
        log.error(f"Failed to load model: {e}")
        raise


def run(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Makes a prediction using the trained ML model."""
    log.info("Received input_data: %s", input_data)

    if MODEL is None:
        raise ValueError("Model is not loaded. Please call init() first.")

    # Load and preprocess the image
    img_path = input_data['image']
    img = load_img(img_path, target_size=(224, 224))  # Adjust size if needed
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # EfficientNetV2 preprocessing

    # Run inference using the loaded model
    try:
        infer = MODEL.signatures["serving_default"]
        input_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        predictions_dict = infer(input_tensor)
    except Exception as e:
        log.error(f"Inference failed: {e}")
        raise

    # Extract predictions (assuming output key is "output_0")
    if "output_0" in predictions_dict:
        predictions = predictions_dict["output_0"].numpy()[0]
    else:
        raise KeyError(f"Unexpected model output keys: {predictions_dict.keys()}")

    log.debug(f'Predictions: {predictions}')

    # Use classifiers directly, expecting a list of class names
    waste_types = input_data['classifiers']
    index = np.argmax(predictions)
    waste_label = waste_types[index]
    accuracy = "{0:.2f}".format(predictions[index] * 100)

    return {"accuracy": accuracy, "label": waste_label}


def sample() -> Dict[str, Any]:
    """Returns a sample input dictionary with dummy image and classifiers."""
    return {
        "image": full_path("sample_image.jpg"),  # Replace with a real sample image path
        "classifiers": ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    }


if __name__ == "__main__":
    init()
    sample_input = sample()
    print(sample_input)
    print(run(sample_input))
