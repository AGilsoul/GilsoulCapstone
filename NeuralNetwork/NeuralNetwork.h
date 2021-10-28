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
    NeuralNetwork(int numLayers, vector<int> neurons, double learningRate);
    void train(vector<vector<double>> input, vector<double> desiredResult);
    static double sigmoid(double input);
    static double sigmoidDeriv(double input);
    static void printVector(vector<double> input);

    struct Neuron {
        Neuron() { bias = 0; }
        vector<double> weights;
        double bias;
        vector<double> prevInputs;
        double prevOutput;
        double calculate(vector<double> input);
    };

private:
    void initializeWeights(int numWeights, Neuron& newN);
    vector<double> forwardProp(vector<double> input);
    void backProp();
    static double costFunction(NeuralNetwork::Neuron input, double desired);
    static double derivWeight(NeuralNetwork::Neuron input, double desired, int index);
    static double derivBias(NeuralNetwork::Neuron input, double desired);
    static double derivInput(NeuralNetwork::Neuron input, double desired);

    vector<vector<Neuron>> layers;
    double learningRate;
    std::mt19937_64 rng;

};

