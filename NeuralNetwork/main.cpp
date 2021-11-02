#include <iostream>
#include <ostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <windows.h>
#include "NeuralNetwork.h"

using std::cout;
using std::endl;
using std::cin;
using std::string;
using std::ios;
using std::ifstream;

void readCancerFile(vector<vector<double>>& testData, vector<vector<double>>& expected, string fileName);
void readBankFile(vector<vector<double>>& data, vector<vector<double>>& expected);


int main() {
    double learningRate = 0.01;
    double momentum = 0.9;
    double numLayers = 2;
    double splitRatio = 0.45;
    vector<int> neuronCounts = {40, 2};
    int iterations = 100;
    string fileName = "Breast_Cancer.csv";
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    vector<vector<double>> data, expected;

    SetConsoleTextAttribute(hConsole, 15);
    cout << endl << "Neural Network Prediction of Malignancy in Breast Tumors" << endl;
    cout << "********************************************************" << endl << endl;
    cout << "Constructing Neural Network with " << numLayers - 1 << " hidden layer(s), learning rate of " << learningRate << ", and momentum of " << momentum << "..." << endl;
    NeuralNetwork net(numLayers, neuronCounts, learningRate, momentum);
    SetConsoleTextAttribute(hConsole, 10);
    cout << "Network construction successful!" << endl << endl;

    SetConsoleTextAttribute(hConsole, 15);
    cout << "Reading data from " << fileName << "..." << endl;
    readCancerFile(data, expected, fileName);
    //readBankFile(data, expected);
    SetConsoleTextAttribute(hConsole, 10);
    cout << "Data collected!" << endl << endl;

    SetConsoleTextAttribute(hConsole, 15);
    cout << "Normalizing data..." << endl;
    net.normalize(data);
    SetConsoleTextAttribute(hConsole, 10);
    cout << "Data normalized!" << endl << endl;

    SetConsoleTextAttribute(hConsole, 15);
    cout << "Splitting data with a training:test ratio of "<< splitRatio * 100 << ":" << (1 - splitRatio) * 100 << "..." << endl;
    auto trainData = net.vectorSplit(data, 0, ceil(data.size() * splitRatio));
    auto testData = net.vectorSplit(data, ceil(data.size() * splitRatio), data.size() - 1);
    auto trainExpected = net.vectorSplit(expected, 0, ceil(expected.size() * splitRatio));
    auto testExpected = net.vectorSplit(expected, ceil(expected.size() * splitRatio), expected.size() - 1);
    SetConsoleTextAttribute(hConsole, 10);
    cout << "Data split!" << endl << endl;

    SetConsoleTextAttribute(hConsole, 15);
    cout << "Training with " << trainData.size() << " data points over " << iterations << " iteration(s)..." << endl;
    net.train(trainData, trainExpected, iterations);
    SetConsoleTextAttribute(hConsole, 10);
    cout << "Model training complete!" << endl << endl;

    SetConsoleTextAttribute(hConsole, 15);
    cout << "Testing with " << testData.size() << " data points..." << endl;
    SetConsoleTextAttribute(hConsole, 10);
    double testResult = net.test(testData, testExpected);
    cout << "Testing complete!" << endl;
    SetConsoleTextAttribute(hConsole, 15);
    cout << "Percent of correctly identified tumors (malignant/benign): " << testResult << "%" << endl;
    cout << endl;
    return 0;
}


void readCancerFile(vector<vector<double>>& testData, vector<vector<double>>& expected, string fileName) {
    //strings to be used for reference and assignment of values when reading the file and assigning to the string list sList
    string id, diagnosis, radMean, texMean, perMean, areaMean, smoothMean, compMean, concMean, concPointMean, symMean, fracMean, radSE, texSE, perSE, areaSE, smoothSE, compSE, concSE, concPointSE, symSE, fracSE, radWorst, perWorst, areaWorst, smoothWorst, compWorst, concWorst, concPointWorst, symWorst, fracWorst;
    string sList[] = {id, diagnosis, radMean, texMean, perMean, areaMean, smoothMean, compMean, concMean, concPointMean, symMean, fracMean, radSE, texSE, perSE, areaSE, smoothSE, compSE, concSE, concPointSE, symSE, fracSE, radWorst, perWorst, areaWorst, smoothWorst, compWorst, concWorst, concPointWorst, symWorst, fracWorst};
    //Reads from the file "Breast_Cancer.csv"
    ifstream fin(fileName, ios::in);
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
                    //cout << "M" << endl;
                    result.push_back(1);
                    result.push_back(0);
                }
                else {
                    //cout << "B" << endl;
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

void readBankFile(vector<vector<double>>& data, vector<vector<double>>& expected) {
    auto rng = std::default_random_engine {};
    rng.seed(time(nullptr));
    vector<vector<double>> allData;
    string waveVariance, waveSkew, waveCurtosis, imageEntropy, id;
    string sList[] = {waveVariance, waveSkew, waveCurtosis, imageEntropy, id};
    ifstream fin("data_banknote_authentication.csv", ios::in);
    int listSize = 5;
    while (!fin.eof()) {
        vector<double> tempData;
        for (int i = 0; i < listSize; i++) {
            if (i != listSize - 1) {
                std::getline(fin, sList[i], ',');
            }
            else {
                std::getline(fin, sList[i], '\n');
            }
            tempData.push_back(stod(sList[i]));
        }
        allData.push_back(tempData);
    }

    shuffle(begin(allData), end(allData), rng);
    for (int vec = 0; vec < allData.size(); vec++) {
        vector<double> newData;
        vector<double> newExpect;
        for (int val = 0; val < allData[vec].size(); val++) {
            if (val != allData[vec].size() - 1) {
                newData.push_back(allData[vec][val]);
            }
            else {
                if (allData[vec][val] == 0) {
                    newExpect.push_back(allData[vec][val]);
                    newExpect.push_back(1);
                }
                else {
                    newExpect.push_back(allData[vec][val]);
                    newExpect.push_back(0);
                }

            }
        }
        data.push_back(newData);
        expected.push_back(newExpect);
    }
    fin.close();
}