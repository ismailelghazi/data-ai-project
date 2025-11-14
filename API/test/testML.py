import numpy as np
import tensorflow as tf


def test_tensorflow_import():
    """Check TensorFlow is installed and working."""
    assert tf.__version__ is not None
    assert callable(tf.constant)


def test_image_shape():
    """Check that a generated image has the correct shape."""
    img = np.random.rand(48, 48).astype("float32")
    assert img.shape == (48, 48)


def preprocess_image(img):
    """Simple preprocessing: reshape and normalize."""
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=(0, -1))  # shape -> (1,48,48,1)
    return img


def test_preprocess_image():
    """Check that our preprocessing outputs valid data."""
    img = np.random.rand(48, 48)
    processed = preprocess_image(img)

    # Verify shape
    assert processed.shape == (1, 48, 48, 1)

    # Values must be between 0 and 1
    assert processed.min() >= 0
    assert processed.max() <= 1
