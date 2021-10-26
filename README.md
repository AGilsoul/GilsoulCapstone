# gilsoulcapstone
My Senior Capstone project/MIT Maker Portolfio, a collection of machine learning algorithms coded from scratch.

KNearestNeighbors:
Uses Euclidean distance between data points to classify unknown points based on the classes of data points nearest to them. Can be computationally expensive with high amounts of data, but can reach high accuracy, averages ~96% with the breast cancer dataset with k = 30.

KMeansClustering:
Similar to KNN, but finds clusters of points and places centroids in the middle, then comparing unkown points to the centroids for classification rather than every single point. In the case of the breast cancer dataset, rather than comparing unkown points to 400 other points, they are just compared to two centroids, one for malignant and one for benign. Much faster than KNN, but oftentimes less accurate.

LinearRegression:
Fits a line, plane, or hyperplane to a set of data for predicting future data.

