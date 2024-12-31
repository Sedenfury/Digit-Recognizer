
As a beginner in machine learning, it was a bit confusing which method(bcuz I dont know much) to use for this PS.
And I dont know what exactly is required for a digit recognizer

So...., I started with some basics which i knew/was learning:
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

I know syntax and other basic stuff in python(thank GOD i decided to do something useful in 1st sem)
Then i had to see which libraries and functions are generally used.


A digit recognizer using the k-Nearest Neighbors (k-NN) algorithm on the MNIST dataset.
The MNIST dataset consists of 70,000 images of handwritten digits (0–9), each of size 28x28 pixels.


The steps involved are:
1. Loading the MNIST dataset.
Now, normally i would use loasd csv file using pandas library
But i came across this while looking for a solution for the PS
fetch_openml('mnist_784', version=1):
   Fetches the MNIST dataset from OpenML, a popular repository for machine learning 
   datasets.
  mnist.data: Contains the feature matrix
  mnist.target: Contains the labels (0–9) for the digits in the dataset.
  k-NN require the target labels to be numeric

3. Preprocessing the data (normalization and train-test split).


4. Training a k-NN classifier, where 'k' is the number of neighbors considered.
5. Evaluating the classifier on the test set using accuracy and a detailed classification report.
