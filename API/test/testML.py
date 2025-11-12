import os
import numpy as np
import tensorflow as tf
import pytest


@pytest.fixture
def dummy_model():
    """Create a minimal CNN model for testing."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input((48, 48, 1)),
        tf.keras.layers.Conv2D(1, 3, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(7, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def test_save_and_load_model(dummy_model):
    """Ensure model saves and loads correctly."""
    path = ".ML\models\emotion_model.h5"

    dummy_model.save(path)
    assert os.path.exists(path), "Model file not created"

    loaded = tf.keras.models.load_model(path)
    assert loaded is not None, "Model failed to load"

    os.remove(path)


def test_prediction_format(dummy_model):
    """Check model prediction shape and probability sum."""
    test_img = np.random.rand(1, 48, 48, 1)
    pred = dummy_model.predict(test_img)

    assert pred.shape == (1, 7), "Prediction shape incorrect"
    assert np.isclose(np.sum(pred), 1.0), "Probabilities do not sum to 1"
