"""
Script de détection et prédiction d'émotions
Utilise Haar Cascade pour détecter les visages et un modèle CNN pour prédire les émotions

Usage:
    python detect_and_predict.py <chemin_vers_image>
"""

import cv2
import numpy as np
from tensorflow import keras
import sys
import os

# Chemins vers les modèles
MODEL_PATH = 'API/ML/models/emotion_model.h5'
CASCADE_PATH = 'API/ML/models/haarcascade_frontalface_default.xml'

# Charger le modèle CNN
try:
    model = keras.models.load_model(MODEL_PATH)
    print("✓ Modèle CNN chargé avec succès")
except Exception as e:
    print(f"✗ Erreur lors du chargement du modèle: {e}")
    sys.exit(1)

# Charger Haar Cascade
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    print("✗ Erreur: Haar Cascade non trouvé")
    sys.exit(1)
print("✓ Haar Cascade chargé avec succès")

# Classes d'émotions (ordre exact du modèle)
emotions = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Mapping pour l'affichage en français
emotions_fr = {
    'angry': 'En colère',
    'disgusted': 'Dégoûté',
    'fearful': 'Peur',
    'happy': 'Heureux',
    'neutral': 'Neutre',
    'sad': 'Triste',
    'surprised': 'Surpris'
}


def detect_and_predict(image_path):
    """
    Détecte le visage dans une image et prédit l'émotion

    Args:
        image_path (str): Chemin vers l'image à analyser

    Returns:
        None (affiche le résultat)
    """
    # Vérifier si le fichier existe
    if not os.path.exists(image_path):
        print(f"✗ Erreur: Image non trouvée à {image_path}")
        return

    # Lire l'image
    img = cv2.imread(image_path)
    if img is None:
        print("✗ Erreur: Impossible de charger l'image")
        return

    print(f"\n📷 Analyse de l'image: {image_path}")
    print(f"   Dimensions: {img.shape[1]}x{img.shape[0]} pixels")

    # Convertir en niveaux de gris
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Détecter les visages
    print("\n🔍 Détection des visages...")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        print("✗ Aucun visage détecté dans l'image")
        return

    print(f"✓ {len(faces)} visage(s) détecté(s)")

    # Pour chaque visage détecté
    for idx, (x, y, w, h) in enumerate(faces, 1):
        print(f"\n--- Visage #{idx} ---")
        print(f"   Position: x={x}, y={y}")
        print(f"   Taille: {w}x{h} pixels")

        # Extraire la région du visage
        face = gray[y:y+h, x:x+w]

        # Redimensionner à 48x48 (taille attendue par le modèle)
        face_resized = cv2.resize(face, (48, 48))

        # Normaliser les valeurs (0-255 -> 0-1)
        face_normalized = face_resized / 255.0

        # Reshape pour le modèle: (1, 48, 48, 1)
        face_input = face_normalized.reshape(1, 48, 48, 1)

        # Prédiction
        print("   🧠 Prédiction en cours...")
        prediction = model.predict(face_input, verbose=0)
        emotion_idx = np.argmax(prediction)
        emotion = emotions[emotion_idx]
        confidence = prediction[0][emotion_idx] * 100

        # Afficher le résultat
        print(f"   ✓ Émotion détectée: {emotion} ({emotions_fr[emotion]})")
        print(f"   ✓ Confiance: {confidence:.2f}%")

        # Dessiner un rectangle vert autour du visage
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Ajouter le texte avec l'émotion et la confiance
        text = f"{emotion}: {confidence:.1f}%"
        cv2.putText(img, text, (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Afficher l'image avec les annotations
    print("\n📊 Affichage du résultat...")
    print("   (Appuyez sur n'importe quelle touche pour fermer)")
    cv2.imshow('Detection et Prediction d\'Emotions', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("✓ Analyse terminée")


def main():
    """Point d'entrée du script"""
    if len(sys.argv) < 2:
        print("Usage: python detect_and_predict.py <chemin_vers_image>")
        print("\nExemple:")
        print("  python detect_and_predict.py test_images/happy_face.jpg")
        sys.exit(1)

    image_path = sys.argv[1]
    detect_and_predict(image_path)


if __name__ == "__main__":
    main()
