//
// Created by agils on 10/19/2021.
//

#pragma once

#include <vector>
#include <memory>
#include <iostream>

using std::vector;
using std::cout;
using std::endl;


class NeuralNetwork {
public:
    struct Neuron {
        Neuron() { bias = 0; }
        vector<double> weights;
        double bias;
        vector<double> prevInputs;
        vector<vector<double>> avgWeightChange;
        vector<double> avgBiasChange;
        double calculate(vector<double> input);
        void calcWeightChange(double desired);
    };

    //Everything public for testing purposes
    NeuralNetwork(int numLayers, vector<int> neurons, double learningRate);
    void train(vector<vector<double>> input, vector<double> desiredResult);
    static double sigmoid(double input);
    static void printVector(vector<double> input);

    void initializeWeights(int numWeights, Neuron& newN);
    static double costFunction(NeuralNetwork::Neuron input, double desired);
    static double derivWeight(NeuralNetwork::Neuron input, double desired, int index);
    static double derivBias(NeuralNetwork::Neuron input, double desired);
    static double derivInput(NeuralNetwork::Neuron input, double desired);
    vector<double> forwardProp(vector<double> input);
    void backProp();




private:




    vector<vector<Neuron>> layers;
    double learningRate;
    //vector<vector<double>> weightMatrix;

};

