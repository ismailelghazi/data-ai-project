import urllib.request
import os

# Créer le dossier models s'il n'existe pas
os.makedirs('models', exist_ok=True)

# URL du fichier Haar Cascade
url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"

# Chemin de destination
output_path = "models/haarcascade_frontalface_default.xml"

# Télécharger
print("Téléchargement de Haar Cascade...")
urllib.request.urlretrieve(url, output_path)
print(f"Fichier téléchargé: {output_path}")

# Vérifier
if os.path.exists(output_path):
    file_size = os.path.getsize(output_path)
    print(f"Taille du fichier: {file_size} bytes")
else:
    print("Erreur lors du téléchargement")