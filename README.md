# GilsoulCapstone
My Senior Capstone project/MIT Maker Portolfio, a collection of machine learning algorithms coded from scratch.

KNearestNeighbors:
Uses Euclidean distance between data points to classify unknown points based on the classes of data points nearest to them. Can be computationally expensive with high amounts of data, but can reach high accuracy, averages ~96% with the breast cancer dataset with k = 30.

KMeansClustering:
Similar to KNN, but finds clusters of points and places centroids in the middle, then comparing unkown points to the centroids for classification rather than every single point. In the case of the breast cancer dataset, rather than comparing unkown points to 400 other points, they are just compared to two centroids, one for malignant and one for benign. Much faster than KNN, but oftentimes less accurate.

LinearRegression:
Fits a line, plane, or hyperplane to a set of data for predicting future data. Adjustments to the line/plane coefficients and bias are determined using gradient descent and partial derivatives with respect to each coefficient and bias. Currently has three sample data runs, the default uses two different variables to generate a plane for predicting the y-value of each set of variables.

NeuralNetwork: Creates a Neural Network that adjusts the weights of connections between structs called neurons to determine probabilities, such as the probability of a written character being a certain number, or of a cat being in a picture. Currently, it is used on the same dataset that is used by the K-Nearest Neighbors, which contains over 500 different tumors, which are classified as either malignant or benign, but I am now training it on a handwritten digit dataset provided by MNIST. While this algorithm takes longer to train than the others, it is one of, if not the most accurate of all my algorithms that I have written. In addition to this, once trained, it can make predictions extremely quickly.
