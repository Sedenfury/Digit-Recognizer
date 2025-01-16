
I started with some basics which i knew/was learning:
1. Supervised and unsupervised learning{basic understanding of classification,regression,clustering,dimensionality,reduction}
2. Simple intro(definitions) to Features:{overfitting,selection, scaling, how it is a part of deep learning}
3. Some linear algebra{L1,L2 norms,weighted euclidean distance,Mahalanobis Distance}
4. Learning with prototype
5. Nearest Neighbors{k-NN,ϵ-NN}
6. Cross Validation
7. Decision Tree{entropy, Info gain}
8. Simple Linear Regression
9. Optimization

Hence i decided to choose a method between Learning with prototype and Nearest Neighbors

First i tried LwP and then kNN.
kNN gave slightly better accuracy.

A digit recognizer using the k-Nearest Neighbors (k-NN) algorithm on the MNIST dataset.
The MNIST dataset consists of 70,000 images of handwritten digits (0–9), each of size 28x28 pixels.


The steps involved are:
1. Loading the MNIST dataset.                                                          
Now, normally i would use load csv file using pandas library
But i came across this while looking for a solution for the PS
fetch_openml('mnist_784', version=1):
   Fetches the MNIST dataset from OpenML, a popular repository for machine learning 
   datasets.
  mnist.data: Contains the feature matrix
  mnist.target: Contains the labels (0–9) for the digits in the dataset.
  k-NN require the target labels to be numeric

2. Preprocessing the data.                                                                 
  Normalize pixel values to [0, 1]
  train test split
   test_size=0.2 : 20% of the data to the test set and 80% to the training set.
   
3. Apply PCA for dimensionality reduction
4. Automate k-NN tuning with GridSearchCV
5. Test the best model
6. Visualize predictions
