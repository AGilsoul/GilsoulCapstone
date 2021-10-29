//
// Created by Alex Gilsoul on 10/19/2021.
//

#pragma once

#include <vector>
#include <memory>
#include <iostream>
#include <cmath>
#include <random>
#include <chrono>

using std::vector;
using std::cout;
using std::endl;
using std::time;


class NeuralNetwork {
public:
    struct Neuron {
        Neuron() { bias = 0; }
        vector<double> weights;
        double bias;
        vector<double> prevInputs;
        double output;
        double delta;
        double calculate(vector<double> input);
    };

    NeuralNetwork(int numLayers, vector<int> neurons, double learningRate);
    void train(vector<vector<double>> input, vector<vector<double>> allResults, int iterations);
    static double sigmoid(double input);
    static double sigmoidDeriv(double input);
    vector<double> forwardProp(vector<double> input);
    double derivWeight(Neuron* curN, int index, double expected);
    double derivBias(Neuron* curN, double expected);
    double finalGradient(Neuron* curN, double expected);
    double hiddenGradient(Neuron* curN, int nIndex, vector<Neuron*> nextLayer, vector<double> nextDeltas);
    double weightDerivative(double neuronError, double prevNeuron);
    static void printVector(vector<double> input);
    vector<vector<double>> vectorSplit(vector<vector<double>> vec, int start, int fin);
    double test(vector<vector<double>> testData, vector<vector<double>> testLabel);


private:
    void initializeWeights(int numWeights, Neuron* newN);
    void backProp();

    vector<vector<Neuron*>> layers;
    double learningRate;
    std::mt19937_64 rng;

};

