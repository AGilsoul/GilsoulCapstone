#include <iostream>
#include <ostream>
#include <fstream>
#include "NeuralNetwork.h"

using std::cout;
using std::endl;
using std::cin;
using std::string;
using std::ios;
using std::ifstream;

void readFile(vector<vector<double>>& testData, vector<vector<double>>& expected);


int main() {
    double learningRate;
    //cout << "Enter learning rate: ";
    //cin >> learningRate;

    //Creates NeuralNetwork "net" with 3 layers, 2 neurons in layer 1, 3 in layer 2, 1 in layer 3
    //sets learning rate to 0.01 (not used yet)
    NeuralNetwork net(4, {16, 16, 4, 2}, 0.00001);
    vector<vector<double>> data;
    vector<vector<double>> expected;
    readFile(data, expected);
    auto trainData = net.vectorSplit(data, 0, ceil(data.size() * 0.8));
    auto testData = net.vectorSplit(data, ceil(data.size() * 0.8), data.size() - 1);
    auto trainExpected = net.vectorSplit(expected, 0, ceil(expected.size() * 0.8));
    auto testExpected = net.vectorSplit(expected, ceil(expected.size() * 0.8), expected.size() - 1);

    //vector<double> expected = {1.0, 0.0, 0.0};
    //Creates a 2d Vector of inputs to test, just one double vector with 3 inputs
    //<vector<double>> testData = {{0.7, 0.2, 0.0}};
    //still not really used yet, no functioning backprop yet
    //auto result = net.forwardProp(testData[0]);
    //cout << "Expected: ";
    //NeuralNetwork::printVector(expected);
    //cout << endl;
    //NeuralNetwork::printVector(result);
    //cout << endl;
    net.train(trainData, trainExpected, 1);
    cout << net.test(testData, testExpected) << endl;

    //result = net.forwardProp(testData[0]);
    //NeuralNetwork::printVector(testData[0]);
    //cout << endl;
    //NeuralNetwork::printVector(testData[1]);
    //NeuralNetwork::printVector(result);
    return 0;
}


void readFile(vector<vector<double>>& testData, vector<vector<double>>& expected) {
    //strings to be used for reference and assignment of values when reading the file and assigning to the string list sList
    string id, diagnosis, radMean, texMean, perMean, areaMean, smoothMean, compMean, concMean, concPointMean, symMean, fracMean, radSE, texSE, perSE, areaSE, smoothSE, compSE, concSE, concPointSE, symSE, fracSE, radWorst, perWorst, areaWorst, smoothWorst, compWorst, concWorst, concPointWorst, symWorst, fracWorst;
    string sList[] = {id, diagnosis, radMean, texMean, perMean, areaMean, smoothMean, compMean, concMean, concPointMean, symMean, fracMean, radSE, texSE, perSE, areaSE, smoothSE, compSE, concSE, concPointSE, symSE, fracSE, radWorst, perWorst, areaWorst, smoothWorst, compWorst, concWorst, concPointWorst, symWorst, fracWorst};
    //Reads from the file "Breast_Cancer.csv"
    ifstream fin("Breast_Cancer.csv", ios::in);
    vector<string> labels;

    int listSize = sizeof(sList) / sizeof(sList[0]);
    while (!fin.eof()) {
        vector<double> dData;
        vector<double> result;
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
            else if (i == 1) {
                if (sList[i] == "M") {
                    result.push_back(1);
                    result.push_back(0);
                }
                else {
                    result.push_back(0);
                    result.push_back(1);
                }
            }
        }

        expected.push_back(result);
        testData.push_back(dData);
    }
    fin.close();
}