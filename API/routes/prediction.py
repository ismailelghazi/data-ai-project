from fastapi import APIRouter, File, UploadFile, Depends, HTTPException
from sqlalchemy.orm import Session
import cv2
import numpy as np
from tensorflow import keras
import os
import json

from core.database import get_db
from models.prediction import Prediction
from schemas.prediction import PredictionResponse, HistoryResponse

router = APIRouter()

# --- Paths to ML files ---
ML_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.', 'ML')
MODEL_PATH = os.path.join(ML_DIR, 'models', 'emotion_model.h5')
CASCADE_PATH = os.path.join(ML_DIR, 'models', 'haarcascade_frontalface_default.xml')
EMOTIONS_PATH = os.path.join(ML_DIR, 'models', 'emotions.json')

# --- Load model, cascade, and emotions ---
model = keras.models.load_model(MODEL_PATH)
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

if os.path.exists(EMOTIONS_PATH):
    with open(EMOTIONS_PATH, 'r') as f:
        emotions = json.load(f)
else:
    emotions = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# --- Prediction endpoint ---
@router.post("/predict_emotion", response_model=PredictionResponse)
async def predict_emotion(file: UploadFile = File(...), db: Session = Depends(get_db)):
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            raise HTTPException(status_code=400, detail="No face detected")

        # Take first face
        x, y, w, h = faces[0]
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48)) / 255.0
        face = face.reshape(1, 48, 48, 1)

        # Predict emotion
        pred = model.predict(face, verbose=0)
        emotion_idx = np.argmax(pred)
        emotion = emotions[emotion_idx]
        confidence = float(pred[0][emotion_idx])

        # Save to database
        new_pred = Prediction(emotion=emotion, confidence=confidence)
        db.add(new_pred)
        db.commit()
        db.refresh(new_pred)

        return PredictionResponse(emotion=emotion, confidence=confidence, message="Prediction successful")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

# --- History endpoint ---
@router.get("/history", response_model=HistoryResponse)
def get_history(db: Session = Depends(get_db)):
    predictions = db.query(Prediction).order_by(Prediction.created_at.desc()).all()
    return HistoryResponse(count=len(predictions), predictions=predictions)
