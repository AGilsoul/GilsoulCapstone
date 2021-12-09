#include <iostream>
#include <ostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <chrono>
#include <windows.h>
#include "NeuralNetwork.h"

using std::cout;
using std::endl;
using std::cin;
using std::string;
using std::ios;
using std::ifstream;
using namespace std::chrono;

void test_cancer_config();
void test_mnist_config();
void cancer_config();
void cancer_minibatch_config();
void mnist_config();
void readMnistFile(vector<vector<double>>& testData, vector<vector<double>>& expected);
void readCancerFile(vector<vector<double>>& testData, vector<vector<double>>& expected, string fileName);

int main() {
    //configuration that loads a pre-trained neural network for breast tumors
    //test_cancer_config();

    //configuration that trains a neural network using mini-batch gd instead of stochastic gd
    //cancer_minibatch_config();

    //configuration that trains and tests a neural network on breast tumors
    //cancer_config();

    //configuration that trains and tests a neural network on handwritten digits
    //WARNING: TAKES A VERY LONG TIME, JUST LOAD THE PRE-TRAINED NETWORK
    //mnist_config();

    //configuration that loads pre-trained neural network for digit recognition, and retrains with mini-batch gradient descent
    test_mnist_config();

    return 0;
}

void test_cancer_config() {
    double splitRatio = 0.6;
    //best with 200
    int iterations = 50;
    string fileName = "Breast_Cancer.csv";
    vector<vector<double>> data, expected;
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);

    SetConsoleTextAttribute(hConsole, 15);
    cout << endl << "Neural Network Prediction of Malignancy in Breast Tumors" << endl;
    cout << "********************************************************" << endl << endl;
    cout << "Loading Neural Network..." << endl;
    NeuralNetwork net("nn_cancer_config.csv");
    SetConsoleTextAttribute(hConsole, 10);
    cout << "Network construction successful!" << endl << endl;

    SetConsoleTextAttribute(hConsole, 15);
    cout << "Reading data from " << fileName << "..." << endl;
    readCancerFile(data, expected, fileName);
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
    cout << "Testing with " << testData.size() << " data points..." << endl;
    SetConsoleTextAttribute(hConsole, 10);
    double testResult = net.test(testData, testExpected);
    cout << "Testing complete!" << endl;
    SetConsoleTextAttribute(hConsole, 15);
    cout << "Percent of correctly identified tumors (malignant/benign): " << testResult << "%" << endl;
    cout << endl;

    SetConsoleTextAttribute(hConsole, 15);
    cout << "Training with " << trainData.size() << " data points over " << iterations << " iteration(s)..." << endl;
    net.train(trainData, trainExpected, iterations, true, "nn_cancer_config.csv");
    SetConsoleTextAttribute(hConsole, 10);
    cout << "Model training complete!" << endl << endl;

    SetConsoleTextAttribute(hConsole, 15);
    cout << "Testing with " << testData.size() << " data points..." << endl;
    SetConsoleTextAttribute(hConsole, 10);
    testResult = net.test(testData, testExpected);
    cout << "Testing complete!" << endl;
    SetConsoleTextAttribute(hConsole, 15);
    cout << "Percent of correctly identified tumors (malignant/benign): " << testResult << "%" << endl;
    cout << endl;
}

void test_mnist_config() {
    //best with 200
    //int iterations = 200;
    double splitRatio = 0.5;
    string fileName = "mnist_train_config.csv";
    vector<vector<double>> data, expected;
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);

    SetConsoleTextAttribute(hConsole, 15);
    cout << endl << "Neural Network Prediction of Handwritten Digits" << endl;
    cout << "********************************************************" << endl << endl;
    cout << "Loading neural network from " << fileName << "..." << endl;
    NeuralNetwork net(fileName);
    SetConsoleTextAttribute(hConsole, 10);
    cout << "Network construction successful!" << endl << endl;

    SetConsoleTextAttribute(hConsole, 15);
    cout << "Reading data from mnist_train.csv..." << endl;
    readMnistFile(data, expected);
    SetConsoleTextAttribute(hConsole, 10);
    cout << "Data collected!" << endl << endl;
    SetConsoleTextAttribute(hConsole, 15);
    cout << "Normalizing " << data.size() << " data points for " << data[0].size() << " categories..." << endl;
    net.normalize(data, {0, 255});
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
    cout << "Testing with " << testData.size() << " data points..." << endl;
    SetConsoleTextAttribute(hConsole, 10);
    double testResult = net.test(testData, testExpected);
    cout << "Testing complete!" << endl;
    SetConsoleTextAttribute(hConsole, 15);
    cout << "Percent of correctly identified digits: " << testResult << "%" << endl;
    cout << endl << endl;

    cout << "Mini-batch Training Analysis" << endl;
    cout << "********************************************************" << endl << endl;
    net.resetWeights(trainData.size());
    cout << "Pre-training accuracy: " << net.test(testData, testExpected);
    cout << endl << "Training..." << endl;
    net.setLR(0.001);
    net.setMomentum(0.9);
    net.trainMiniBatch(trainData, trainExpected, 50, 2048);
    SetConsoleTextAttribute(hConsole, 10);
    cout << "Training Complete!" << endl << endl;
    SetConsoleTextAttribute(hConsole, 15);
    cout << "Post-training accuracy: " << net.test(testData, testExpected) << endl;

}

void mnist_config() {
    double learningRate = 0.01;
    double momentum = 0.9;
    //number of layers excluding input layer
    double splitRatio = 0.75;
    //neuron counts for hidden and output layers
    vector<int> neuronCounts = {100, 10};
    //best with 200
    int iterations = 200;
    vector<vector<double>> data, expected;
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);

    SetConsoleTextAttribute(hConsole, 15);
    cout << endl << "Neural Network Prediction of Handwritten Digits" << endl;
    cout << "********************************************************" << endl << endl;
    cout << "Constructing Neural Network with " << neuronCounts.size() - 1 << " hidden layer(s), learning rate of " << learningRate << ", and momentum of " << momentum << "..." << endl;
    NeuralNetwork net(neuronCounts, learningRate, momentum);
    SetConsoleTextAttribute(hConsole, 10);
    cout << "Network construction successful!" << endl << endl;

    SetConsoleTextAttribute(hConsole, 15);
    cout << "Reading data from mnist_train.csv..." << endl;
    readMnistFile(data, expected);
    SetConsoleTextAttribute(hConsole, 10);
    cout << "Data collected!" << endl << endl;
    SetConsoleTextAttribute(hConsole, 15);
    cout << "Normalizing " << data.size() << " data points for " << data[0].size() << " categories..." << endl;
    net.normalize(data, {0, 255});
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
    net.train(trainData, trainExpected, iterations, true, "mnist_train_config.csv");
    SetConsoleTextAttribute(hConsole, 10);
    cout << "Model training complete!" << endl << endl;

    SetConsoleTextAttribute(hConsole, 15);
    cout << "Testing with " << testData.size() << " data points..." << endl;
    SetConsoleTextAttribute(hConsole, 10);
    double testResult = net.test(testData, testExpected);
    cout << "Testing complete!" << endl;
    SetConsoleTextAttribute(hConsole, 15);
    cout << "Percent of correctly identified digits: " << testResult << "%" << endl;
    cout << endl;
}

void cancer_config() {
    double learningRate = 0.01;
    double momentum = 0.9;
    //number of layers excluding input layer
    double splitRatio = 0.6;
    //neuron counts for hidden and output layers
    vector<int> neuronCounts = {60, 2};
    //best with 200
    int iterations = 50;
    string fileName = "Breast_Cancer.csv";
    vector<vector<double>> data, expected;
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    SetConsoleTextAttribute(hConsole, 15);
    cout << endl << "Neural Network Prediction of Malignancy in Breast Tumors" << endl;
    cout << "********************************************************" << endl << endl;
    cout << "Constructing Neural Network with " << neuronCounts.size() - 1 << " hidden layer(s), learning rate of " << learningRate << ", and momentum of " << momentum << "..." << endl;
    NeuralNetwork net(neuronCounts, learningRate, momentum);
    SetConsoleTextAttribute(hConsole, 10);
    cout << "Network construction successful!" << endl << endl;

    SetConsoleTextAttribute(hConsole, 15);
    cout << "Reading data from " << fileName << "..." << endl;
    readCancerFile(data, expected, fileName);
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
    net.train(trainData, trainExpected, iterations, true, "nn_cancer_config.csv");
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
}

void cancer_minibatch_config() {
    double learningRate = 0.01;
    double momentum = 0.9;
    //number of layers excluding input layer
    double splitRatio = 0.6;
    //neuron counts for hidden and output layers
    vector<int> neuronCounts = {30, 2};
    //best with 200
    int iterations = 200;
    string fileName = "Breast_Cancer.csv";
    vector<vector<double>> data, expected;
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    SetConsoleTextAttribute(hConsole, 15);
    cout << endl << "Neural Network Prediction of Malignancy in Breast Tumors" << endl;
    cout << "********************************************************" << endl << endl;
    cout << "Constructing Neural Network with " << neuronCounts.size() - 1 << " hidden layer(s), learning rate of " << learningRate << ", and momentum of " << momentum << "..." << endl;
    NeuralNetwork net(neuronCounts, learningRate, momentum);
    SetConsoleTextAttribute(hConsole, 10);
    cout << "Network construction successful!" << endl << endl;

    SetConsoleTextAttribute(hConsole, 15);
    cout << "Reading data from " << fileName << "..." << endl;
    readCancerFile(data, expected, fileName);
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
    net.trainMiniBatch(trainData, trainExpected, iterations, 32);
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
}

void readMnistFile(vector<vector<double>>& testData, vector<vector<double>>& expected) {
    //784 data columns, one label column at beginning
    //strings to be used for reference and assignment of values when reading the file and assigning to the string list sList
    string sList[785];
    for (unsigned int i = 0; i < 785; i++) {
        string temp = "";
        sList[i] = temp;
    }
    //Reads from the file "mnist_train.csv"
    ifstream fin("mnist_train.csv", ios::in);
    vector<string> labels;
    int listSize = sizeof(sList) / sizeof(sList[0]);
    while (!fin.eof()) {
        vector<double> dData;
        vector<double> result;
        for (unsigned int i = 0; i < listSize; i++) {
            if (i != listSize - 1) {
                getline(fin, sList[i], ',');
            }
            else {
                getline(fin, sList[i], '\n');
            }

            if (i != 0) {
                dData.push_back(stod(sList[i]));
            }
            else {
                for (int x = 0; x < 10; x++) {
                    if (stod(sList[i]) == x) {
                        result.push_back(1);
                    }
                    else {
                        result.push_back(0);
                    }
                }
            }
        }

        expected.push_back(result);
        testData.push_back(dData);
    }
    fin.close();
}

void readCancerFile(vector<vector<double>>& testData, vector<vector<double>>& expected, string fileName) {
    //strings to be used for reference and assignment of values when reading the file and assigning to the string list sList
    string sList[30];
    for (unsigned int i = 0; i < 30; i++) {
        sList[i] = "";
    }
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