import os
import numpy as np
import tensorflow as tf
import pytest
import tempfile
import shutil


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


@pytest.fixture
def temp_model_dir():
    """Create a temporary directory for model testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup after test
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


def test_save_and_load_model(dummy_model, temp_model_dir):
    """Ensure model saves and loads correctly."""
    path = os.path.join(temp_model_dir, "test_emotion_model.h5")

    # Save the model
    dummy_model.save(path)
    assert os.path.exists(path), "Model file not created"

    # Load the model
    loaded = tf.keras.models.load_model(path)
    assert loaded is not None, "Model failed to load"

    # Verify model structure
    assert len(loaded.layers) == len(dummy_model.layers), "Model structure mismatch"


def test_prediction_format(dummy_model):
    """Check model prediction shape and probability sum."""
    test_img = np.random.rand(1, 48, 48, 1)
    pred = dummy_model.predict(test_img)

    assert pred.shape == (1, 7), "Prediction shape incorrect"
    assert np.isclose(np.sum(pred), 1.0), "Probabilities do not sum to 1"
