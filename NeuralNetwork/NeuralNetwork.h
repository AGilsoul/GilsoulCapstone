//
// Created by Alex Gilsoul on 10/19/2021.
//

#pragma once

#include <vector>
#include <memory>
#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>

using std::vector;
using std::cout;
using std::endl;
using std::time;
using std::ofstream;
using std::ifstream;
using std::string;
using std::ios;


class NeuralNetwork {
public:
    struct Neuron {
        Neuron() { bias = 0; }
        vector<double> weights;
        double bias;
        vector<double> prevInputs;
        vector<double> prevGradients;
        double prevBias;
        double output;
        double delta;
        double calculate(vector<double> input);
    };

    NeuralNetwork(vector<int> neurons, double learningRate = 0.01, double momentum = 0.9);
    NeuralNetwork(string fileName = "");
    void normalize(vector<vector<double>>& input, vector<double> minMaxRange = {}, bool save = false, string fileName = "");
    void loadData(string fileName);
    void train(vector<vector<double>> input, vector<vector<double>> allResults, int iterations, bool save = false, string fileName = "");
    void trainMiniBatch(vector<vector<double>> input, vector<vector<double>> allResults, int iterations, int epochs, bool save = false, string fileName = "");
    vector<double> forwardProp(vector<double> input);
    static void printVector(vector<double> input);
    vector<vector<double>> vectorSplit(vector<vector<double>> vec, int start, int fin);
    double test(vector<vector<double>>& testData, vector<vector<double>>& testLabel);
    vector<double> predict(vector<double> unknownP);
    void resetWeights(int inputCount);
    void setMomentum(double m) {
        this->momentum = m;
    }
    void setLR(double lr) {
        this->learningRate = lr;
    }

private:

    vector<double> predictTest(vector<double> unknownP);
    double sigmoid(double input);
    double sigmoidDeriv(double input);
    double relu(double input);
    double reluDeriv(double input);
    void initializeWeights(int numWeights, Neuron* newN, double numOut);
    double finalGradient(Neuron* curN, double expected);
    double hiddenGradient(Neuron* curN, int nIndex, vector<Neuron*> nextLayer, vector<double> nextDeltas);
    double weightDerivative(double neuronError, double prevNeuron);
    vector<double> sortVector(vector<double> vec);
    bool saveData(string fileName);

    vector<vector<Neuron*>> layers;
    vector<vector<double>> conversionRates;
    bool conversions = false;
    bool loadedData = false;
    double learningRate;
    double momentum;
    std::mt19937_64 rng;

};

