import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt


# 1. Load the MNIST dataset
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data, mnist.target

# Convert labels to integers
y = y.astype(int)

# 2. Preprocess the data
print("Preprocessing data...")
X = X / 255.0  # Normalize pixel values to [0, 1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Apply PCA for dimensionality reduction
print("Applying PCA...")
pca = PCA(n_components=0.95)  # Retain 95% of the variance
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print(f"Original number of features: {X.shape[1]}")
print(f"Reduced number of features: {X_train_pca.shape[1]}")

# 4. Automate k-NN tuning with GridSearchCV
print("Performing GridSearchCV for hyperparameter tuning...")
param_grid = {
    'n_neighbors': [1, 3, 5, 7, 9, 11],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=3, scoring='accuracy', verbose=1)
grid_search.fit(X_train_pca, y_train)

# Best parameters and accuracy
best_knn = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validated accuracy: {grid_search.best_score_ * 100:.2f}%")

# 5. Test the best model
print("Testing the best model...")
y_pred = best_knn.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test accuracy with best parameters: {accuracy * 100:.2f}%")

# Optional: Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 6. Visualize predictions
def visualize_predictions(X_test, y_test, y_pred, num_samples=10):
    plt.figure(figsize=(10, 4))
    for i in range(num_samples):
        plt.subplot(2, num_samples // 2, i + 1)
        plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
        plt.title(f"True: {y_test[i]}\nPred: {y_pred[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

print("Visualizing predictions...")
visualize_predictions(X_test[:10], y_test[:10], y_pred[:10])
