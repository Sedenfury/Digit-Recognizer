The steps involved are:
1. Loading the MNIST dataset.                                                          
fetch_openml('mnist_784', version=1):
   Fetches the MNIST dataset from OpenML, a popular repository for machine learning 
   datasets.
  X = mnist.data: Contains the feature matrix, 1D vector of length 784(28x28)
  y = mnist.target: Contains the labels (0–9) for the digits in the dataset.
  k-NN require the target labels to be numeric, y.astype(int) ensures the labels are treated as integers.

2. Preprocessing the data.                                                                 
  Normalize pixel values to [0, 1] (originally between 0 and 255)
  train test split
   test_size=0.2 : 20% of the data to the test set and 80% to the training set.
   
3. Apply PCA for dimensionality reduction
PCA is a dimensionality reduction technique that transforms data into a smaller set of uncorrelated variables called principal components.
It does this by identifying the directions (or axes) of maximum variance in the data and projecting the data onto these axes.
(Basically PCA is rotation of axes which represented our original variables to new axes that represent variables that account for variation in decresing order)

n_components=0.95 means PCA will select enough components to retain 95% of the total variance in the data.

The fit_transform function:
Fits: Learns the principal components (directions of maximum variance) from the training data X_train.
Transforms: Projects the training data onto these principal components to reduce its dimensionality.
The result is the transformed X_train_pca, which has fewer features but retains most of the important information.

transform(X_test)
same transformation is applied to the test data X_test.
to maintain same dimensions of both test and train datasets.

Visualizing PCA Variance: This plots the cumulative variance explained by each PCA component to show how much information is retained as the number of components increases.



4. Automate k-NN tuning with GridSearchCV
The fit function tests all combinations of hyperparameters using cross-validation on X_train_pca and y_train.
est_estimator_: The k-NN model with the best combination of hyperparameters.
best_params_: The exact values of n_neighbors and metric that yielded the highest accuracy.
best_score_: The cross-validated accuracy of the best model.

5. Test the best model
Uses the best k-NN model (from GridSearchCV) to predict labels for the test data (X_test_pca).
Measures test accuracy using accuracy_score.


6. Classification Report
Prints precision, recall, and F1-score for each digit class, providing a detailed analysis of the model's performance.


7. Confusion Matrix
Displays a confusion matrix that shows how many samples of each true class were classified correctly or incorrectly.
Highlights specific classes that are frequently confused with others.

8. Prediction Probabilities
Predicts the probability distribution for the first 10 samples, showing how confident the model is for each class.




