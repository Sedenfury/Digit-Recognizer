
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


# 1. Load the MNIST dataset
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data, mnist.target

# Convert labels to integers
y = y.astype(np.int)

# 2. Preprocess the data
print("Preprocessing data...")
X = X / 255.0  # Normalize pixel values to [0, 1]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Initialize the k-NN classifier
k = 3  # Number of neighbors
knn = KNeighborsClassifier(n_neighbors=k)

# 4. Train the k-NN model
print("Training the k-NN classifier...")
knn.fit(X_train, y_train)

# 5. Test the model
print("Testing the model...")
y_pred = knn.predict(X_test)

# 6. Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

#Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
