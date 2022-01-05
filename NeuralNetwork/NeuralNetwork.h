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
    /***
     * Neuron Struct
     * Contains weights and bias for one neuron in the neural network
     * Calculates output using output from previous layer as input
     */
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

    /***
     * Constructor for a new neural network
     * @param neurons vector containing the number of neurons for each hidden layer and the output layer
     * @param learningRate learning rate of the model
     * @param momentum momentum of the model
     */
    NeuralNetwork(vector<int> neurons, double learningRate = 0.01, double momentum = 0.9, bool verbose = false, int barSize = 70);

    /***
     * Constructor for loading in a new neural network
     * @param fileName name of the configuration file to be loaded
     */
    NeuralNetwork(string fileName = "", bool verbose = false, int barSize = 70);

    /***
     * Normalizes input data
     * If every data characteristic uses the same range, minMaxRange can be specified to speed up the process
     * This is useful for processing data from inputs such as greyscale pictures, where each characteristic is 0-255
     * @param input input data
     * @param minMaxRange vector containing minimum and maximum data values for all categories
     * @param save set to true when the user wants to save a configuration file for the normalization ranges
     * @param fileName file name of the normalization range file
     */
    void normalize(vector<vector<double>>& input, vector<double> minMaxRange = {});
    void loadData(string fileName);
    void train(vector<vector<double>> trainInput, vector<vector<double>> trainResults, vector<vector<double>> valInput, vector<vector<double>> valResults, int maxIterations);
    void train(vector<vector<double>> input, vector<vector<double>> allResults, int iterations);
    void trainMiniBatch(vector<vector<double>> input, vector<vector<double>> allResults, int iterations, int batchSize);
    vector<double> forwardProp(vector<double> input, double chanceDropout);
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
    void setDropOut(double rate) {
        this->dropOutRate = rate;
    }
    void setEarlyStopping(int numStop) {
        this->earlyStopping = numStop;
    }
    bool saveModel(string fileName) {
        cout << "Saving data..." << endl;
        bool saveSuccess = saveData(fileName);
        if (saveSuccess) {
            cout << "Data saved successfully as " << fileName << endl;
            return true;
        }
        else {
            cout << "Failed to save data" << endl;
            return false;
        }
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
    double weightDerivative(double neuronDelta, double input);
    vector<double> sortVector(vector<double> vec);
    bool saveData(string fileName);
    void progressBar(double curVal, double goal);

    vector<vector<Neuron*>> layers;
    vector<vector<double>> conversionRates;
    bool conversions = false;
    bool loadedData = false;
    bool verbose = true;
    int barSize = 70;
    double learningRate;
    double momentum;
    double dropOutRate = 1.0;
    int earlyStopping = -1;
    std::mt19937_64 rng;
};

