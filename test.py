import cv2
import mediapipe as mp
import pandas as pd
import os

# Initialiser MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Paramètres du modèle
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Dossier contenant les images
IMAGE_FOLDER = "dataset/"  # Modifie ce chemin vers ton dataset
OUTPUT_CSV = "face_landmarks.csv"

# Liste pour stocker les résultats
data = []

# Lire toutes les images du dataset
for filename in os.listdir(IMAGE_FOLDER):
    if filename.endswith((".jpg", ".png", ".jpeg")):
        image_path = os.path.join(IMAGE_FOLDER, filename)
        image = cv2.imread(image_path)
        
        # Convertir en RGB (obligatoire pour Mediapipe)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

            # Afficher l'image avec les points
            cv2.imshow("Face Landmarks", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        # Vérifier si un visage a été détecté
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extraire les coordonnées X, Y, Z des 468 landmarks
                landmarks = []
                for i, landmark in enumerate(face_landmarks.landmark):
                    landmarks.extend([landmark.x, landmark.y, landmark.z])  # Ajouter X, Y, Z

                # Ajouter au dataset
                data.append([filename] + landmarks)

# Création du DataFrame pandas
columns = ["image"] + [f"{coord}{i}" for i in range(478) for coord in ["X", "Y", "Z"]]

df = pd.DataFrame(data, columns=columns)

# Sauvegarde en CSV
df.to_csv(OUTPUT_CSV, index=False)
print(f"Fichier CSV généré : {OUTPUT_CSV}")

# Lire le fichier CSV généré
df_read = pd.read_csv("./face_landmarks.csv")

# Afficher les premières lignes du fichier
print(df_read.head())

