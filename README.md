# GilsoulCapstone
My Senior Capstone project/MIT Maker Portolfio, a collection of machine learning algorithms coded from scratch.

KNearestNeighbors:
Uses Euclidean distance between data points to classify unknown points based on the classes of data points nearest to them. Can be computationally expensive with high amounts of data, but can reach high accuracy, averages ~96% with the breast cancer dataset with k = 30.

KMeansClustering:
Similar to KNN, but finds clusters of points and places centroids in the middle, then comparing unkown points to the centroids for classification rather than every single point. In the case of the breast cancer dataset, rather than comparing unkown points to 400 other points, they are just compared to two centroids, one for malignant and one for benign. Much faster than KNN, but oftentimes less accurate.

LinearRegression:
Fits a line, plane, or hyperplane to a set of data for predicting future data. Adjustments to the line/plane coefficients and bias are determined using gradient descent and partial derivatives with respect to each coefficient and bias. Currently has three sample data runs, the default uses two different variables to generate a plane for predicting the y-value of each set of variables.

NeuralNetwork: Creates a Neural Network that adjusts the weights of connections between structs called neurons to determine probabilities, such as the probability of a written character being a certain number, or of a cat being in a picture. There are four different run configurations in the main method, two methods for two datasets (breast tumor malignancy recognition and handwritten digit recognition). The "test_<datasetname>_config" functions load a pre-trained neural network, and test it, while the <datasetname>_config create a new neural network, train it, and save the data in a csv file. The neural networks can be trained on the breast cancer dataset in a reasonable amount of time (< 1 minute), but for the MNIST handwritten digit dataset, it can take at least an hour to train, so I recommend that you load the pre-trained model. On both datasets, the networks' accuracy sits comfortably in the 97%-100% range, with 100% accuracy not beign uncommon. 
