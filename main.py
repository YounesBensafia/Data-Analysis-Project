import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = pd.read_csv('data.csv')

sampled_data = data.sample(n=30, random_state=1)


def distance(x1, y1, x2, y2):
    return -np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def angle(x1, y1, x2, y2, x3, y3):
    a = np.array([x1, y1])
    b = np.array([x2, y2])
    c = np.array([x3, y3])

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return -np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

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

# Configuration du style des graphiques
plt.style.use('seaborn')
sns.set_palette("husl")

# Création d'une figure avec plusieurs sous-graphiques
plt.figure(figsize=(20, 12))

# 1. Distribution des distances
plt.subplot(2, 2, 1)
distance_cols = [col for col in sampled_data.columns if 'distance' in col]
distance_data = sampled_data[distance_cols].melt()
sns.boxplot(x='variable', y='value', data=distance_data)
plt.xticks(rotation=45)
plt.title('Distribution des distances faciales')
plt.xlabel('Mesures')
plt.ylabel('Distance (pixels)')

# 2. Distribution des angles
plt.subplot(2, 2, 2)
angle_cols = [col for col in sampled_data.columns if 'angle' in col]
angle_data = sampled_data[angle_cols].melt()
sns.boxplot(x='variable', y='value', data=angle_data)
plt.xticks(rotation=45)
plt.title('Distribution des angles faciaux')
plt.xlabel('Mesures')
plt.ylabel('Angle (degrés)')

# 3. Corrélation entre les distances des yeux et du nez
plt.subplot(2, 2, 3)
sns.scatterplot(data=sampled_data, x='distance_eyes', y='distance_eye_nose_left', alpha=0.6)
plt.title('Corrélation: Distance des yeux vs Distance œil-nez gauche')
plt.xlabel('Distance entre les yeux')
plt.ylabel('Distance œil-nez gauche')

# 4. Corrélation entre les angles
plt.subplot(2, 2, 4)
sns.scatterplot(data=sampled_data, x='angle_nose_eyes', y='angle_nose_mouth', alpha=0.6)
plt.title('Corrélation: Angle nez-yeux vs Angle nez-bouche')
plt.xlabel('Angle nez-yeux')
plt.ylabel('Angle nez-bouche')

# Ajuster la mise en page
plt.tight_layout()

# Sauvegarder les graphiques
plt.savefig('facial_features_analysis.png', dpi=300, bbox_inches='tight')

# Sauvegarder les données
sampled_data.to_csv('data.csv', index=True)

# Préparation des données pour ACP
features_for_pca = ['distance_eyes', 'distance_eye_nose_left', 'distance_eye_nose_right',
                    'distance_nose_mouth_left', 'distance_nose_mouth_right', 'distance_mouth',
                    'angle_nose_eyes', 'angle_nose_mouth']

# Standardisation des données
scaler = StandardScaler()
data_scaled = scaler.fit_transform(sampled_data[features_for_pca])

# Application de l'ACP
pca = PCA()
pca_result = pca.fit_transform(data_scaled)

# Calcul de la variance expliquée
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# Création d'une nouvelle figure pour l'ACP
plt.figure(figsize=(15, 10))

# 1. Scree plot
plt.subplot(2, 2, 1)
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 'bo-')
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 'ro-')
plt.xlabel('Composantes principales')
plt.ylabel('Ratio de variance expliquée')
plt.title('Scree Plot')
plt.legend(['Variance individuelle', 'Variance cumulée'])

# 2. Biplot des deux premières composantes
plt.subplot(2, 2, 2)
plt.scatter(pca_result[:, 0], pca_result[:, 1])
plt.xlabel(f'PC1 ({explained_variance_ratio[0]:.2%} variance expliquée)')
plt.ylabel(f'PC2 ({explained_variance_ratio[1]:.2%} variance expliquée)')
plt.title('Projection ACP sur PC1 et PC2')

# 3. Contribution des variables
loadings = pca.components_.T
plt.subplot(2, 2, 3)
sns.heatmap(loadings, annot=True, cmap='coolwarm', xticklabels=[f'PC{i+1}' for i in range(loadings.shape[1])],
            yticklabels=features_for_pca)
plt.title('Contribution des variables aux composantes principales')

plt.tight_layout()
plt.savefig('pca_analysis.png', dpi=300, bbox_inches='tight')

# Sauvegarder les données avec les composantes principales
pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])
result_df = pd.concat([sampled_data, pca_df], axis=1)
result_df.to_csv('data_with_pca.csv', index=True) 