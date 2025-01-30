from sklearn.datasets import fetch_lfw_people
data = fetch_lfw_people(min_faces_per_person=1, resize=0.4)
X = data.images  # Shape: (n_samples, 50, 37) -> Matrix form
y = data.target  # Labels

print(X, y)