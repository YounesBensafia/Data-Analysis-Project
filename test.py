import cv2
import mediapipe as mp
import os
import csv
import numpy as np
import pandas as pd

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

IMAGE_FOLDER = "./Celebrity_Faces_Dataset/"
OUTPUT_CSV = "face_landmarks.csv"

if not os.path.exists(IMAGE_FOLDER):
    print(f"❌ Error: Dataset path '{IMAGE_FOLDER}' not found!")
    exit()

with open(OUTPUT_CSV, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    
    header = ["filename"]
    for i in range(478):
        header.extend([f"landmark_{i}_x", f"landmark_{i}_y", f"landmark_{i}_z"])
    csv_writer.writerow(header)
    
    for root, _, files in os.walk(IMAGE_FOLDER):
        print(f"📂 Processing folder: {root}")
        
        for filename in files:
            if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                image_path = os.path.normpath(os.path.join(root, filename))
                print(f"✅ Processing: {image_path}")
                
                label = os.path.basename(root)
                
                image = cv2.imread(image_path)
                if image is None:
                    print(f"❌ Failed to read: {image_path}")
                    continue  
                
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                results = face_mesh.process(image_rgb)
                if not results.multi_face_landmarks:
                    print(f"❌ No face detected in: {filename}")
                    continue 
                
                face_landmarks = results.multi_face_landmarks[0]
                
                row = [filename]
                
                for landmark in face_landmarks.landmark:
                    row.extend([landmark.x, landmark.y, landmark.z])
                
                csv_writer.writerow(row)

print(f"✅ CSV file created successfully: {OUTPUT_CSV}")

try:
    with open(OUTPUT_CSV, 'r') as f:
        line_count = sum(1 for _ in f) - 1 
    print(f"📊 Total images processed: {line_count}")
    print(f"📊 Features per image: {len(header) - 1}")
    df = pd.read_csv(OUTPUT_CSV)
    print("📋 First 5 rows of the CSV file:")
    print(df.head())
except:
    print("❌ Error counting lines in the output file")