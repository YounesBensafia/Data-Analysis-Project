import numpy as np
import pandas as pd

data = pd.read_csv('data.csv')

sampled_data = data.sample(n=30, random_state=1)


def distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def angle(x1, y1, x2, y2, x3, y3):
    a = np.array([x1, y1])
    b = np.array([x2, y2])
    c = np.array([x3, y3])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

sampled_data['distance_eyes'] = sampled_data.apply(lambda row: 
    distance(row['lefteye_x'], row['lefteye_y'], row['righteye_x'], row['righteye_y']), axis=1)

sampled_data['distance_eye_nose_left'] = sampled_data.apply(lambda row: 
    distance(row['lefteye_x'], row['lefteye_y'], row['nose_x'], row['nose_y']), axis=1)

sampled_data['distance_eye_nose_right'] = sampled_data.apply(lambda row: 
    distance(row['righteye_x'], row['righteye_y'], row['nose_x'], row['nose_y']), axis=1)

sampled_data['distance_nose_mouth_left'] = sampled_data.apply(lambda row: 
    distance(row['nose_x'], row['nose_y'], row['leftmouth_x'], row['leftmouth_y']), axis=1)

sampled_data['distance_nose_mouth_right'] = sampled_data.apply(lambda row: 
    distance(row['nose_x'], row['nose_y'], row['rightmouth_x'], row['rightmouth_y']), axis=1)

sampled_data['distance_mouth'] = sampled_data.apply(lambda row: 
    distance(row['leftmouth_x'], row['leftmouth_y'], row['rightmouth_x'], row['rightmouth_y']), axis=1)

sampled_data['angle_nose_eyes'] = sampled_data.apply(lambda row: 
    angle(row['lefteye_x'], row['lefteye_y'], row['nose_x'], row['nose_y'], row['righteye_x'], row['righteye_y']), axis=1)

sampled_data['angle_nose_mouth'] = sampled_data.apply(lambda row: 
    angle(row['leftmouth_x'], row['leftmouth_y'], row['nose_x'], row['nose_y'], row['rightmouth_x'], row['rightmouth_y']), axis=1)

sampled_data.to_csv('data.csv', index=False)



