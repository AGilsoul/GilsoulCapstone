#include <iostream>
#include <ostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <random>
#include <chrono>
#include <windows.h>
#include "../include/NeuralNetwork.h"

using std::cout;
using std::endl;
using std::cin;
using std::string;
using std::ios;
using std::ifstream;
using namespace std::chrono;

void performanceTesting(vector<vector<double>> inputData, vector<vector<double>> inputExpected, int iterations);
void test_mnist_config();
void mnist_config();
void cancer_config();
void energy_config();
void readMnistFile(vector<vector<double>>& testData, vector<vector<double>>& expected);
void readCancerFile(vector<vector<double>>& testData, vector<vector<double>>& expected, string fileName);
void energyFile(vector<vector<double>>& testData, vector<vector<double>>& expected, string fileName);
vector<double> oneHotEncode(int value, int numCategories);
void readClassificationQuantitativeFile(vector<vector<double>>& testData, vector<vector<double>>& expected, string fileName, int numColumns, int targetColumn, int numTargets);


int main() {
    string placeHolder;
    //configuration that loads a pre-trained neural network for breast tumors
    //test_cancer_config();

    //configuration that trains a neural network using mini-batch gd instead of stochastic gd
    //cancer_minibatch_config();

    //configuration that trains and tests a neural network on breast tumors
    //cancer_config();

    //configuration that trains and tests a neural network on handwritten digits
    //mnist_config();

    //configuration that loads pre-trained neural network for digit recognition, and retrains with mini-batch gradient descent
    //test_mnist_config();

    //regression configuration that trains a neural network on residential structure data and predicts cooling load
    //energy_config();

    //regression config for excitation current of synchronous machines
    //synchronous_machine_config();

    //classification config that predicts a class of Cherenkov radiation producing event
    //gamma_ray_config();

    //classification config that predicts whether signals are pulsars or not
    //pulsar_config();


    vector<vector<double>> data, expected;
    string fileName = "mnist_train.csv";
    //readMnistFile(data, expected);
    readClassificationQuantitativeFile(data, expected, fileName, 755, 0, 10);
    //string fileName = "Breast_Cancer.csv";
    //readCancerFile(data, expected, fileName);
    performanceTesting(data, expected, 3);


    cout << "Press x to continue" << endl;
    cin >> placeHolder;
    return 0;
}

void performanceTesting(vector<vector<double>> inputData, vector<vector<double>> inputExpected, int iterations) {
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    cout << endl << "NEURAL NETWORK PERFORMANCE TEST AND COMPARISON" << endl;
    cout << "********************************************************" << endl << endl;
    double avgAccuracies[4] = {0, 0, 0};
    double avgTimes[4] = {0, 0, 0};
    double learningRate = 0.001;
    double momentum = 0.9;
    double dropOutRate = 0.8;
    vector<double> splitRatios = {0.5, 0.2, 0.3};
    //neuron counts for hidden and output layers
    vector<int> neuronCounts = {100, 10};
    //best with 200
    int earlyStopping = 5;
    StopWatch myWatch;
    for (int mCount = 0; mCount < 3; mCount++) {
        for (int i = 0; i < iterations; i++) {
            SetConsoleTextAttribute(hConsole, 15);
            NeuralNetwork net(neuronCounts, learningRate);
            net.setMomentum(momentum);
            net.setEarlyStopping(earlyStopping);
            net.setVerbose(true);
            net.setDropOut(dropOutRate);
            SetConsoleTextAttribute(hConsole, 10);
            SetConsoleTextAttribute(hConsole, 15);
            vector<vector<double>> data = inputData;
            vector<vector<double>> expected = inputExpected;
            net.normalize(data, {0, 255});
            //net.normalize(data);

            vector<int> indexes;
            indexes.reserve(data.size());
            for (int i = 0; i < data.size(); ++i)
                indexes.push_back(i);

            std::random_shuffle(indexes.begin(), indexes.end());
            vector<vector<double>> newData;
            vector<vector<double>> newExpect;
            for (unsigned int i = 0; i < data.size(); i++) {
                newData.push_back(data[indexes[i]]);
                newExpect.push_back(expected[indexes[i]]);
            }
            data = newData;
            expected = newExpect;

            auto allData = net.trainValTestSplit(data, splitRatios);
            auto allLabels = net.trainValTestSplit(expected, splitRatios);
            auto trainData = allData[0];
            auto trainExpected = allLabels[0];
            auto valData = allData[1];
            auto valExpected = allLabels[1];
            auto testData = allData[2];
            auto testExpected = allLabels[2];

            SetConsoleTextAttribute(hConsole, 15);
            cout << "Training model " << mCount + 1 << " | Iteration " << i + 1 << endl;
            if (mCount == 0) {
                myWatch.reset();
                net.trainMiniBatchValidation(trainData, trainExpected, valData, valExpected, 20, 50, 64);
                avgTimes[mCount] += myWatch.elapsed_time();
            }
            else if (mCount == 1) {
                myWatch.reset();
                net.trainWithValidation(trainData, trainExpected, valData, valExpected, 20, 50);
                avgTimes[mCount] += myWatch.elapsed_time();
            }
            else if (mCount == 2) {
                myWatch.reset();
                net.trainWithValidation(trainData, trainExpected, valData, valExpected, 10, 25);
                net.trainMiniBatchValidation(trainData, trainExpected, valData, valExpected, 10, 25, 64);
                avgTimes[mCount] += myWatch.elapsed_time();
            }
            double testResult = net.test(testData, testExpected);
            avgAccuracies[mCount] += testResult;
            SetConsoleTextAttribute(hConsole, 15);
            cout << "Model " << mCount + 1 << " iteration " << i + 1 << " accuracy: " << testResult << "%" << endl;
            cout << endl;
        }
    }
    for (int i = 0; i < 3; i++) {
        cout << "Model " << i + 1 << " avg accuracy: " << avgAccuracies[i]/iterations << "% | avg time: " << avgTimes[i]/iterations << endl;
    }
    cout << endl;
}

void test_mnist_config() {
    //train:test split
    double splitRatio = 0.6;
    string fileName = "mnist_train_config.csv";
    vector<vector<double>> data, expected;

    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    SetConsoleTextAttribute(hConsole, 15);
    cout << endl << "Neural Network Prediction of Handwritten Digits" << endl;
    cout << "********************************************************" << endl << endl;
    cout << "Loading neural network from " << fileName << "..." << endl;
    NeuralNetwork net(fileName, true);
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

    vector<int> indexes;
    indexes.reserve(data.size());
    for (int i = 0; i < data.size(); ++i)
        indexes.push_back(i);

    std::random_shuffle(indexes.begin(), indexes.end());
    vector<vector<double>> newData;
    vector<vector<double>> newExpect;
    for (unsigned int i = 0; i < data.size(); i++) {
        newData.push_back(data[indexes[i]]);
        newExpect.push_back(expected[indexes[i]]);
    }
    data = newData;
    expected = newExpect;

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
    cout << "Pre-training accuracy: " << net.test(testData, testExpected) << "%";
    cout << endl << "Training (This Could Take a Few Minutes)..." << endl;
    net.setLR(0.01);
    net.setMomentum(0.9);
    net.trainMiniBatch(trainData, trainExpected, 40, 32);
    SetConsoleTextAttribute(hConsole, 10);
    cout << "Training Complete!" << endl << endl;
    SetConsoleTextAttribute(hConsole, 15);
    cout << "Post-training validation accuracy: " << net.test(trainData, trainExpected) << "%" << endl;
    cout << "Post-training accuracy: " << net.test(testData, testExpected) << "%" << endl;

}

void mnist_config() {
    double learningRate = 0.001;
    double momentum = 0.9;
    double dropOutRate = 0.8;
    vector<double> splitRatios = {0.6, 0.2, 0.2};
    //neuron counts for hidden and output layers
    vector<int> neuronCounts = {32, 10};
    //best with 200
    int minIterations = 40;
    int maxIterations = 200;
    int earlyStopping = 10;
    vector<vector<double>> data, expected;
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);

    SetConsoleTextAttribute(hConsole, 15);
    cout << endl << "Neural Network Prediction of Handwritten Digits" << endl;
    cout << "********************************************************" << endl << endl;
    cout << "Constructing Neural Network with " << neuronCounts.size() - 1 << " hidden layer(s), learning rate of " << learningRate << endl;
    NeuralNetwork net(neuronCounts, learningRate);
    net.setMomentum(momentum);
    net.setEarlyStopping(earlyStopping);
    net.setVerbose(true);
    net.setDropOut(dropOutRate);
    SetConsoleTextAttribute(hConsole, 10);
    cout << "Network construction successful!" << endl << endl;

    SetConsoleTextAttribute(hConsole, 15);
    cout << "Reading data from mnist_train.csv..." << endl;
    //readRegressionQuantitativeFile(data, expected, "mnist_train.csv", 785, 0);
    readMnistFile(data, expected);
    SetConsoleTextAttribute(hConsole, 10);
    cout << "Data collected!" << endl << endl;
    SetConsoleTextAttribute(hConsole, 15);
    cout << "Normalizing " << data.size() << " data points for " << data[0].size() << " categories..." << endl;
    net.normalize(data, {0, 255});
    SetConsoleTextAttribute(hConsole, 10);
    cout << "Data normalized!" << endl << endl;

    vector<int> indexes;
    indexes.reserve(data.size());
    for (int i = 0; i < data.size(); ++i)
        indexes.push_back(i);

    std::random_shuffle(indexes.begin(), indexes.end());
    vector<vector<double>> newData;
    vector<vector<double>> newExpect;
    for (unsigned int i = 0; i < data.size(); i++) {
        newData.push_back(data[indexes[i]]);
        newExpect.push_back(expected[indexes[i]]);
    }
    data = newData;
    expected = newExpect;

    SetConsoleTextAttribute(hConsole, 15);
    cout << "Splitting data with a training:val:test ratio of "<< splitRatios[0] * 100 << ":" << splitRatios[1] * 100 << ":" << splitRatios[2] * 100 << "..." << endl;
    auto allData = net.trainValTestSplit(data, splitRatios);
    auto allLabels = net.trainValTestSplit(expected, splitRatios);
    auto trainData = allData[0];
    auto trainExpected = allLabels[0];
    auto valData = allData[1];
    auto valExpected = allLabels[1];
    auto testData = allData[2];
    auto testExpected = allLabels[2];
    SetConsoleTextAttribute(hConsole, 10);
    cout << "Data split!" << endl << endl;

    SetConsoleTextAttribute(hConsole, 15);
    cout << "Training with mini-batch supplemented by SGD..." << endl;
    net.trainMiniBatch(trainData, trainExpected, 50, 32);
    net.train(trainData, trainExpected, 20);
    //net.saveModel("mnist_train_config.csv");
    SetConsoleTextAttribute(hConsole, 10);
    cout << "Model training complete!" << endl << endl;
    SetConsoleTextAttribute(hConsole, 15);
    cout << "Validation Accuracy: " << net.test(valData, valExpected) << "%" << endl << endl;

    SetConsoleTextAttribute(hConsole, 15);
    cout << "Testing with " << testData.size() << " data points..." << endl;
    double testResult = net.test(testData, testExpected);
    SetConsoleTextAttribute(hConsole, 10);
    cout << "Testing complete!" << endl;
    SetConsoleTextAttribute(hConsole, 15);
    cout << "Percent of correctly identified digits: " << testResult << "%" << endl;
    cout << endl;
}

void cancer_config() {
    double learningRate = 0.001;
    double momentum = 0.9;
    //double dORate = 0.5;
    //number of layers excluding input layer
    vector<double> splitRatios = {0.6, 0.2, 0.2};
    //neuron counts for hidden and output layers
    vector<int> neuronCounts = {30, 2};
    //best with 200
    int maxIterations = 1000;
    int minIterations = 250;
    int earlyStopping = 10;
    string fileName = "Breast_Cancer.csv";
    vector<vector<double>> data, expected;
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    SetConsoleTextAttribute(hConsole, 15);
    cout << endl << "Neural Network Prediction of Malignancy in Breast Tumors" << endl;
    cout << "********************************************************" << endl << endl;
    cout << "Constructing Neural Network with " << neuronCounts.size() - 1 << " hidden layer(s), learning rate of " << learningRate << ", and momentum of " << momentum << "..." << endl;
    NeuralNetwork net(neuronCounts, learningRate, momentum, true);
    //net.setDropOut(dORate);
    net.setEarlyStopping(earlyStopping);
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

    vector<int> indexes;
    indexes.reserve(data.size());
    for (int i = 0; i < data.size(); ++i)
        indexes.push_back(i);

    std::random_shuffle(indexes.begin(), indexes.end());
    vector<vector<double>> newData;
    vector<vector<double>> newExpect;
    for (unsigned int i = 0; i < data.size(); i++) {
        newData.push_back(data[indexes[i]]);
        newExpect.push_back(expected[indexes[i]]);
    }
    data = newData;
    expected = newExpect;

    SetConsoleTextAttribute(hConsole, 15);
    cout << "Splitting data with a training:validation:test ratio of "<< splitRatios[0] * 100 << ":" << splitRatios[1] * 100 << ":" << splitRatios[2] * 100 << "..." << endl;
    auto allData = net.trainValTestSplit(data, {.6, .2, .2});
    auto allLabels = net.trainValTestSplit(expected, {.6, .2, .2});
    auto trainData = allData[0];
    auto trainExpected = allLabels[0];
    auto valData = allData[1];
    auto valExpected = allLabels[1];
    auto testData = allData[2];
    auto testExpected = allLabels[2];

    SetConsoleTextAttribute(hConsole, 10);
    cout << "Data split!" << endl << endl;

    SetConsoleTextAttribute(hConsole, 15);
    cout << "Training with " << trainData.size() << " data points over " << minIterations << " < x < " << maxIterations << " iteration(s) |  early stopping: " << earlyStopping << endl;
    net.trainWithValidation(trainData, trainExpected, valData, valExpected, minIterations, maxIterations);
    //net.saveModel("nn_cancer_config.csv");
    SetConsoleTextAttribute(hConsole, 10);
    cout << "Model training complete!" << endl;
    SetConsoleTextAttribute(hConsole, 15);
    cout << "Validation Accuracy: " << net.test(valData, valExpected) << "%" << endl << endl;

    SetConsoleTextAttribute(hConsole, 15);
    cout << "Testing with " << testData.size() << " data points..." << endl;
    SetConsoleTextAttribute(hConsole, 10);
    double testResult = net.test(testData, testExpected);
    cout << "Testing complete!" << endl;
    SetConsoleTextAttribute(hConsole, 15);
    cout << "Percent of correctly identified tumors (malignant/benign): " << testResult << "%" << endl;
    cout << endl;
}

void energy_config() {
    double learningRate = 0.001;
    double momentum = 0.0;
    double dropOutRate = 1.0;
    int earlyStopping = 15;
    //number of layers excluding input layer
    vector<double> splitRatios = {0.6, 0.2, 0.2};
    //neuron counts for hidden and output layers
    vector<int> neuronCounts = {15, 15, 1};
    //best with 200
    int maxIterations = 1000;
    int minIterations = 100;
    string fileName = "energyefficiency.csv";
    vector<vector<double>> data, expected;
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    SetConsoleTextAttribute(hConsole, 15);
    cout << endl << "Neural Network Prediction of Cooling Energy Load in Residential Buildings" << endl;
    cout << "********************************************************" << endl << endl;
    cout << "Constructing Neural Network with " << neuronCounts.size() - 1 << " hidden layer(s), learning rate of " << learningRate << ", and momentum of " << momentum << "..." << endl;
    NeuralNetwork net(neuronCounts, learningRate, momentum);
    net.setDropOut(dropOutRate);
    net.setVerbose(true);
    net.setEarlyStopping(earlyStopping);
    SetConsoleTextAttribute(hConsole, 10);
    cout << "Network construction successful!" << endl << endl;

    SetConsoleTextAttribute(hConsole, 15);
    cout << "Reading data from " << fileName << "..." << endl;
    energyFile(data, expected, fileName);
    SetConsoleTextAttribute(hConsole, 10);
    cout << "Data collected!" << endl << endl;

    vector<int> indexes;
    indexes.reserve(data.size());
    for (int i = 0; i < data.size(); ++i)
        indexes.push_back(i);

    std::random_shuffle(indexes.begin(), indexes.end());
    vector<vector<double>> newData;
    vector<vector<double>> newExpect;
    for (unsigned int i = 0; i < data.size(); i++) {
        newData.push_back(data[indexes[i]]);
        newExpect.push_back(expected[indexes[i]]);
    }
    data = newData;
    expected = newExpect;

    SetConsoleTextAttribute(hConsole, 15);
    cout << "Splitting data with a training:validation:test ratio of "<< splitRatios[0] * 100 << ":" << splitRatios[1] * 100 << ":" << splitRatios[2] * 100 << "..." << endl;
    auto allData = net.trainValTestSplit(data, splitRatios);
    auto allTargets = net.trainValTestSplit(expected, splitRatios);
    auto trainData = allData[0];
    auto valData = allData[1];
    auto testData = allData[2];
    auto trainTargets = allTargets[0];
    auto valTargets = allTargets[1];
    auto testTargets = allTargets[2];
    SetConsoleTextAttribute(hConsole, 10);
    cout << "Data split!" << endl << endl;

    SetConsoleTextAttribute(hConsole, 15);
    cout << "Normalizing data..." << endl;
    net.normalize(trainData);
    net.normalize(valData);
    net.normalize(testData);
    SetConsoleTextAttribute(hConsole, 10);
    cout << "Data normalized!" << endl << endl;

    SetConsoleTextAttribute(hConsole, 15);
    cout << "Training with " << trainData.size() << " data points over " << minIterations << " < x < " << maxIterations << " iteration(s) | dropout rate: " << dropOutRate << endl;
    net.trainWithValidation(trainData, trainTargets, valData, valTargets, minIterations, maxIterations);
    SetConsoleTextAttribute(hConsole, 10);
    cout << "Model training complete!" << endl << endl;

    SetConsoleTextAttribute(hConsole, 15);
    cout << "Testing with " << testData.size() << " data points..." << endl;
    SetConsoleTextAttribute(hConsole, 10);
    double testResult = net.test(testData, testTargets);
    cout << "Testing complete!" << endl;
    SetConsoleTextAttribute(hConsole, 15);
    cout << "Validation R^2: " << net.test(valData, valTargets) << endl;
    cout << "Testing R^2: " << testResult << endl << endl;

    vector<double> example = {0.76,661.5,416.5,122.5,7,4,0.4,3};
    vector<vector<double>> feed = {example};
    net.normalize(feed);
    cout << "Predicting Cooling Load of Residential Building Sample 647:" << endl;
    cout << "Relative Compactness: " << example[0] << endl;
    cout << "Surface Area: " << example[1] << " m^2" << endl;
    cout << "Wall Area: " << example[2] << " m^2" << endl;
    cout << "Roof Area: " << example[3] << " m^2" << endl;
    cout << "Overall Height: " << example[4] <<  " m" << endl;
    cout << "Orientation: " << example[5] << " (2:North, 3:East, 4:South, 5:West)" << endl;
    cout << "Glazing Area: " << example[6] << "%" << endl;
    cout << "Glazing Area Distribution: " << example[7] << " (Variance - 1:Uniform, 2:North, 3:East, 4:South, 5:West)" << endl;
    cout << endl << "Predicted Cooling Load: " << net.predict(example)[0] << " kWh/m^2" << endl;

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
    //vector<string> labels;
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

void energyFile(vector<vector<double>>& testData, vector<vector<double>>& expected, string fileName) {
    //strings to be used for reference and assignment of values when reading the file and assigning to the string list sList
    string sList[9];
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

            if (i != 8) {
                dData.push_back(stod(sList[i]));
            }
            else if (i == 8) {
                result = {stod(sList[8])};
            }
        }
        expected.push_back(result);
        testData.push_back(dData);
    }
    fin.close();
}

void readClassificationQuantitativeFile(vector<vector<double>>& testData, vector<vector<double>>& expected, string fileName, int numColumns, int targetColumn, int numTargets) {
    //strings to be used for reference and assignment of values when reading the file and assigning to the string list sList
    string pHolder = "";
    ifstream fin(fileName, ios::in);
    while (!fin.eof()) {
        vector<double> dData;
        vector<double> result;
        for (int i = 0; i < numColumns; i++) {
            if (i != numColumns - 1) {
                getline(fin, pHolder, ',');
            }
            else {
                getline(fin, pHolder, '\n');
            }

            if (i == targetColumn) {
                result = oneHotEncode(stoi(pHolder), numTargets);
            }
            else {
                dData.push_back(stod(pHolder));
            }
        }
        testData.push_back(dData);
        expected.push_back(result);
    }
    fin.close();
}

vector<double> oneHotEncode(int value, int numCategories) {
    vector<double> encoded(numCategories, 0);
    encoded[value] = 1;
    return encoded;
}
