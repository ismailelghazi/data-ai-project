import cv2
import numpy as np
from tensorflow import keras

# Charger le modèle
model = keras.models.load_model('models/emotion_model.h5')

# Charger Haar Cascade
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

# Classes d'émotions (TON ORDRE exact!)
emotions = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

def detect_emotion(image_path):
    # Lire l'image
    img = cv2.imread(image_path)
    
    # Si l'image ne charge pas, arrêter
    if img is None:
        print("Erreur: image pas trouvée")
        return
    
    # Convertir en gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Détecter les visages
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Si pas de visage, dire et arrêter
    if len(faces) == 0:
        print("Pas de visage trouvé")
        return
    
    # Pour chaque visage trouvé
    for (x, y, w, h) in faces:
        # Prendre juste le visage
        face = gray[y:y+h, x:x+w]
        
        # Redimensionner à 48x48
        face = cv2.resize(face, (48, 48))
        
        # Normaliser
        face = face / 255.0
        
        # Reshape pour le modèle
        face = face.reshape(1, 48, 48, 1)
        
        # Prédire
        prediction = model.predict(face, verbose=0)
        emotion_idx = np.argmax(prediction)
        emotion = emotions[emotion_idx]
        confidence = prediction[0][emotion_idx] * 100
        
        # Dessiner rectangle vert
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Écrire le texte
        text = f"{emotion}: {confidence:.1f}%"
        cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        print(f"Emotion: {emotion}, Confiance: {confidence:.2f}%")
    
    # Montrer l'image
    cv2.imshow('Resultat', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Test
image_path = r'c:\Users\DELL\Pictures\Camera Roll\WIN_20251110_09_31_34_Pro.jpg'
detect_emotion(image_path)