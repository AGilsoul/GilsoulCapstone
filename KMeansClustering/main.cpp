#include <iostream>
#include "KMeans.h"
#include <vector>
#include <fstream>
#include <algorithm>
#include <chrono>

using namespace std;
using namespace std::chrono;

vector<KMeans::Point> readFileCancer();

int main() {
    double accuracy = 0;
    double trainAvg = 0;
    double testAvg = 0;
    double result = 0;
    int iterations, k;


    cout << "Enter k number of centroids to test with:";
    cin >> k;
    cout << "Enter number of epochs(iterations) to test data on:";
    cin >> iterations;
    KMeans classifier(k);

    auto rng = default_random_engine {};
    vector<KMeans::Point> patientData = readFileCancer();
    classifier.normalize(patientData);
    cout << endl;
    for (int i = 0; i < iterations; i++) {
        cout << "******************************-Epoch " << i + 1 << "-******************************" << endl;
        rng.seed(time(nullptr));
        //rng.seed(5);
        shuffle(begin(patientData), end(patientData), rng);
        auto trainSplit = classifier.vectorSplit(patientData, 0, ceil(patientData.size() * 0.6));
        auto testSplit = classifier.vectorSplit(patientData, ceil(patientData.size() * 0.6), patientData.size() - 1);
        cout << "training..." << endl;
        auto trainStart = high_resolution_clock::now();
        classifier.train(trainSplit);
        auto trainEnd = high_resolution_clock::now();
        auto trainDuration = duration_cast<microseconds>(trainEnd - trainStart).count() / trainSplit.size();
        trainAvg += trainDuration;
        cout << "converged" << endl;
        auto testStart = high_resolution_clock::now();
        result = classifier.runTest(testSplit, trainSplit);
        auto testEnd = high_resolution_clock::now();
        auto testDuration = duration_cast<microseconds>(testEnd - testStart).count() /  testSplit.size();
        testAvg += testDuration;
        accuracy += result;
        cout << "Accuracy of epoch " << i + 1 << ": " << result << "%" << endl;
    }
    cout << endl << "Percent of correctly identified data points with " << k << " centroids: " << accuracy / iterations << "%" << endl;
    cout << "Average training time: " << trainAvg / iterations << "ms" << endl;
    cout << "Average testing time: " << testAvg / iterations << "ms" << endl;

    return 0;
}

vector<KMeans::Point> readFileCancer() {
    vector<KMeans::Point> points;
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
    points = KMeans::convertData(labels, doubleData);
    fin.close();
    return points;
}