#pragma once

#include <iostream>
#include <utility>
#include <vector>
#include <cmath>

using std::vector;
using std::string;
using std::ostream;
using std::ceil;
using std::pow;

//KNNClassifier class definition
class KNNClassifier {
public:
    //Public Point data struct, contains coordinates of point (data) and a string label
    struct Point {
        Point(string label, vector<double> data): label(std::move(label)), data(std::move(data)) {}
        string label;
        vector<double> data;
    };
    //Default constructor
    KNNClassifier();
    //Overloaded constructor, initializes k value
    explicit KNNClassifier(int k);
    //Normalization method, takes a vector of Points and normalizes all data values to be within the range [0,1]
    void normalize(vector<Point>& input);
    //Vector splitting method, returns a vector containing only from a specified range from the original vector
    static vector<Point> vectorSplit(vector<Point> input, int start, int fin);
    //Tests the accuracy of the classifier, returns % Points correctly identified
    double runTest(const vector<Point>& testData);
    //Trains the classifier, assigns training data points to currentData variable
    void train(vector<Point> input);
    //Predicts the label of a given vector of data containing doubles
    string predictData(vector<double> input);
    //Given a vector of string labels, and a two-dimensional vector of doubles, converts data to Point structs
    vector<Point> convertData(vector<string> labels, vector<vector<double>> input);
    //Assigns an integer value to k
    void setK(int k);

private:
    //Searches a vector of strings for a specific string
    static int getIndex(vector<string> vec, const string& data);
    //Given a vector of doubles, sorts from least to greatest
    static vector<double> sortVector(vector<double> vec);
    //Returns the Euclidean distance between a vector of points and each Point in a vector containing Points
    static vector<double> getDistances(vector<double> coords, vector<Point> input);
    //Predicts the label of a given vector of data containing doubles, using a given vector of Points, typically currentData
    string predict(vector<double> data, vector<Point> points) const;
    //Specified "k" value
    int k;
    //Data the classifier is trained on
    vector<Point> currentData;
};


