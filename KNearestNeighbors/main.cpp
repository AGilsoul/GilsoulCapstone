#include <iostream>
#include <vector>
#include <iomanip>
#include <sstream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include "KNNClassifier.h"

using namespace std;

vector<KNNClassifier::Point> readFile(KNNClassifier);

//Driver for example testing file

int main() {
    double accuracy = 0;
    double result;
    int iterations, k;

    cout << "Enter number of nearest points/neighbors (K) to analyze: ";
    cin >> k;
    cout << endl;
    cout << "Enter number of epochs(iterations) to test data on:";
    cin >> iterations;
    cout << endl;

    //creates a randomizing engine to randomize data split
    auto rng = default_random_engine {};
    //Creates a KNNClassifier object, called classifier, with value of k
    KNNClassifier classifier(k);
    //Initialize patientData Point vector from readFile() function
    auto patientData = readFile(classifier);

    //normalizes data for all Points in patientData vector
    classifier.normalize(patientData);

    //For every user-specified iteration
    for (int i = 0; i < iterations; i++) {
        //Sets the seed of the randomizing engine depending on the current time
        rng.seed(time(nullptr));
        //Shuffles the patientData vector using the randomizing engine
        shuffle(begin(patientData), end(patientData), rng);
        //Does an 80/20 split on the patientData vector, 80% of the data for training, 20% for testing
        auto trainSplit = classifier.vectorSplit(patientData, 0, ceil(patientData.size() * 0.8));
        auto testSplit = classifier.vectorSplit(patientData, ceil(patientData.size() * 0.8), patientData.size() - 1);
        //Trains the classifier using the training data
        classifier.train(trainSplit);
        //Gets the result from testing the classifier on the testing data
        result = classifier.runTest(testSplit);
        accuracy += result;
        cout << "Accuracy of epoch " << i + 1 << ": " << result << "%" << endl;
    }
    cout << endl << "Percent of correctly identified tumors (Malignant/Benign): " << accuracy / iterations << "%" << endl;
    return 0;
}

//Reads data from .csv file, and converts it to a vector of points using the convertData method in KNNClassifier
vector<KNNClassifier::Point> readFile(KNNClassifier classifier) {
    vector<KNNClassifier::Point> points;
    //strings to be used for reference and assignment of values when reading the file and assigning to the string list sList
    string id, diagnosis, radMean, texMean, perMean, areaMean, smoothMean, compMean, concMean, concPointMean, symMean, fracMean, radSE, texSE, perSE, areaSE, smoothSE, compSE, concSE, concPointSE, symSE, fracSE, radWorst, perWorst, areaWorst, smoothWorst, compWorst, concWorst, concPointWorst, symWorst, fracWorst;
    string sList[] = {id, diagnosis, radMean, texMean, perMean, areaMean, smoothMean, compMean, concMean, concPointMean, symMean, fracMean, radSE, texSE, perSE, areaSE, smoothSE, compSE, concSE, concPointSE, symSE, fracSE, radWorst, perWorst, areaWorst, smoothWorst, compWorst, concWorst, concPointWorst, symWorst, fracWorst};
    //Reads from the file "Breast_Cancer.csv"
    ifstream fin("Breast_Cancer.csv", ios::in);
    vector<string> labels;
    vector<vector<double>> doubleData;

    int listSize = sizeof(sList) / sizeof(sList[0]);
    while (!fin.eof()) {
        vector<double> dData;
        for (int i = 0; i < listSize; i++) {
            if (i != listSize - 1) {
                getline(fin, sList[i], ',');
            }
            else {
                getline(fin, sList[i], '\n');
            }

            if (i != 1 && i != 0) {
                dData.push_back(stod(sList[i]));
            }
        }

        labels.push_back(sList[1]);
        doubleData.push_back(dData);
    }
    points = classifier.convertData(labels, doubleData);
    fin.close();
    return points;
}