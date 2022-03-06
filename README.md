# GilsoulCapstone
My Senior Capstone project/MIT Maker Portolfio, a collection of machine learning algorithms coded from scratch.

NeuralNetwork: Creates a Neural Network that adjusts the weights of connections between structs called neurons to determine probabilities, such as the probability of a written character being a certain number. There are four different run configurations in the main method, two methods for two datasets (breast tumor malignancy recognition and handwritten digit recognition). The "test_<datasetname>_config" functions load a pre-trained neural network, and tests it, while the <datasetname>_config create a new neural network, train it, and save the data in a csv file. The neural networks can be trained on the breast cancer dataset in a reasonable amount of time (< 1 minute), but for the full MNIST handwritten digit dataset, it can take at least an hour to train using stochastic gradient descent (dataset includes 60,000 samples with 784 features each), so I recommend that you load the pre-trained model instead. The network can also use mini-batch gradient descent, and while the accuracy suffers a little, the training time is reduced to under 10 minutes in most cases, a huge improvement time wise (it is still a work in progres, I am fine tuning to find the best training configuration for mini-batch for optimal accuracy). On both datasets, the networks accuracy sits comfortably in the 97%-100% range using stochastic gradient descent, with 100% accuracy not being uncommon. the network can also perform regression, and currently does so with high accuracy (R^2 > 0.9) on two datasets, one for prediciting the heating/cooling load of residential structures, and the other for predicting the excitation current of synchronous machines.

KNearestNeighbors:
Uses Euclidean distance between data points to classify unknown points based on the classes of data points nearest to them. Can be computationally expensive with high amounts of data, but can reach high accuracy, averages ~96% with the breast cancer dataset with k = 30.

KMeansClustering:
Similar to KNN, but finds clusters of points and places centroids in the middle, then comparing unkown points to the centroids for classification rather than every single point. In the case of the breast cancer dataset, rather than comparing unkown points to 400 other points, they are just compared to two centroids, one for malignant and one for benign. Much faster than KNN, but oftentimes less accurate.

LinearRegression:
Fits a line, plane, or hyperplane to a set of data for predicting future data. Adjustments to the line/plane coefficients and bias are determined using stochastic gradient descent and partial derivatives with respect to each coefficient and bias. Currently has three sample data runs, the default uses two different variables to generate a plane for predicting the y-value of each set of variables.


