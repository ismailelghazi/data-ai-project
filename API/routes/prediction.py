from fastapi import APIRouter, File, UploadFile, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import cv2
import numpy as np
from tensorflow import keras

from core.database import get_db
from models.prediction import Prediction
from schemas.prediction import PredictionResponse, HistoryResponse

router = APIRouter()

# Charger le modèle et Haar Cascade
model = keras.models.load_model('models/emotion_model.h5')
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
emotions = ['angry', 'happy', 'sad', 'surprised']  # Ajustez selon vos classes

@router.post("/predict_emotion", response_model=PredictionResponse)
async def predict_emotion(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Prédit l'émotion à partir d'une image"""
    
    try:
        # Lire l'image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Image invalide")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Détecter le visage
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            raise HTTPException(status_code=400, detail="Aucun visage détecté")
        
        # Prendre le premier visage
        (x, y, w, h) = faces[0]
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (48, 48))
        face_normalized = face_resized / 255.0
        face_input = face_normalized.reshape(1, 48, 48, 1)
        
        # Prédiction
        prediction = model.predict(face_input)
        emotion_idx = np.argmax(prediction)
        emotion = emotions[emotion_idx]
        confidence = float(prediction[0][emotion_idx])
        
        # Sauvegarder dans la base de données
        new_prediction = Prediction(emotion=emotion, confidence=confidence)
        db.add(new_prediction)
        db.commit()
        db.refresh(new_prediction)
        
        return PredictionResponse(
            emotion=emotion,
            confidence=confidence,
            message="Prédiction réussie!"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur serveur: {str(e)}")

@router.get("/history", response_model=HistoryResponse)
def get_prediction_history(db: Session = Depends(get_db)):
    """Récupère l'historique des prédictions"""
    
    predictions = db.query(Prediction).order_by(Prediction.created_at.desc()).all()
    
    return HistoryResponse(
        count=len(predictions),
        predictions=predictions
    )