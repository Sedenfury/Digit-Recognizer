A digit recognizer using the k-Nearest Neighbors (k-NN) algorithm on the MNIST dataset.
The MNIST dataset consists of 70,000 images of handwritten digits (0–9), each of size 28x28 pixels.

The steps involved are:
1. Loading the MNIST dataset.
2. Preprocessing the data (normalization and train-test split).
3. Training a k-NN classifier, where 'k' is the number of neighbors considered.
4. Evaluating the classifier on the test set using accuracy and a detailed classification report.

Why k-NN?
- k-NN is a simple yet powerful algorithm that works by finding the 'k' closest neighbors in the feature space.
- The predicted class is determined by majority voting among these neighbors.

Why choose k=3?
- A small value of 'k' helps the model capture local patterns in the data.
- Choosing an odd value like 3 avoids ties in majority voting.
- This script uses k=3 as a starting point, but this value can be tuned further for better performance.# Digit-Recognizer
