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
#include <thread>
#include <iomanip>
#include "alg_stopwatch.h"
#include <windows.h>

using std::vector;
using std::cout;
using std::endl;
using std::time;
using std::ofstream;
using std::ifstream;
using std::string;
using std::ios;
using std::shared_ptr;
using std::make_shared;


class NeuralNetwork {
public:
    /***
     * Neuron Struct
     * Contains weights and bias for one neuron in the neural network
     * Calculates output using output from previous layer as input
     */
    struct Neuron {
        // Default constructor, sets the bias to 0
        Neuron(): bias(0) {}
        // vector storing neuron weights
        vector<double> weights;
        // bias value for the neuron
        double bias;
        // vector storing the previous inputs to the neuron
        vector<double> prevInputs;
        // vector storing the previous gradients applied to the neuron
        vector<double> prevGradients;
        // previous bias value for the neuron
        double prevBias;
        // stores the output of the neuron
        double output;
        // stores the output of the neuron after the activation function is applied
        double activatedOutput;
        // stores the delta value of the neuron
        double delta;
        // output calculation function
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
    void perTest(vector<vector<double>> input, vector<vector<double>> allResults, vector<vector<double>> testIn, vector<vector<double>> testResults, int iterations, string fOut);
    void trainWithValidation(vector<vector<double>> trainInput, vector<vector<double>> trainResults, vector<vector<double>> valInput, vector<vector<double>> valResults, int minIterations, int maxIterations);
    void train(vector<vector<double>> input, vector<vector<double>> allResults, int iterations);
    void trainMiniBatch(vector<vector<double>> input, vector<vector<double>> allResults, int iterations, int batchSize);
    void trainMiniBatchValidation(vector<vector<double>> trainInput, vector<vector<double>> trainResults, vector<vector<double>> valInput, vector<vector<double>> valResults, int minIterations, int maxIterations, int batchSize);    vector<double> forwardProp(vector<double> input, double chanceDropout);
    static void printVector(vector<double> input);
    vector<vector<double>> vectorSplit(vector<vector<double>> vec, int start, int fin);
    vector<vector<vector<double>>> trainValTestSplit(vector<vector<double>> vec, vector<double> splitRatios);
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
    void setVerbose(bool verbose) {
        this->verbose = verbose;
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

    bool writeResults(string fileName, vector<double> primKey, vector<double> results, vector<double> runtime);
    void _perTest(vector<vector<double>> input, vector<vector<double>> allResults, vector<vector<double>> testIn, vector<vector<double>> testResults, int iterations, string fOut);
    void _trainWithValidation(vector<vector<double>> trainInput, vector<vector<double>> trainResults, vector<vector<double>> valInput, vector<vector<double>> valResults, int minIterations, int maxIterations);
    void _train(vector<vector<double>> input, vector<vector<double>> allResults, int iterations);
    void _trainMiniBatch(vector<vector<double>> input, vector<vector<double>> allResults, int iterations, int batchSize);
    void _trainMiniBatchValidation(vector<vector<double>> trainInput, vector<vector<double>> trainResults, vector<vector<double>> valInput, vector<vector<double>> valResults, int minIterations, int maxIterations, int batchSize);
    vector<double> predictTest(vector<double> unknownP);
    vector<double> softmax();
    double relu(double input) const;
    double reluDeriv(double input) const;
    void initializeWeights(int numWeights, shared_ptr<Neuron> newN, double numOut);
    void resetGradients();
    vector<double> finalSoftmaxGradient(vector<double> targets);
    double finalLinearGradient(shared_ptr<Neuron> curN, double expected) const;
    double hiddenGradient(shared_ptr<Neuron> curN, int nIndex, vector<shared_ptr<Neuron>> nextLayer, vector<double> nextDeltas) const;
    double weightDerivative(double neuronDelta, double input) const;
    vector<double> vectorMinMax(vector<double> vec);
    bool saveData(string fileName);
    void progressBar();
    void printBar(int curVal, int goal, int barWidth, StopWatch watch, int startTime);
    vector<double> rSquared(vector<vector<double>> predicted, vector<vector<double>> target);
    double crossEntropy(vector<vector<double>> predicted, vector<vector<double>> target);


    vector<vector<shared_ptr<Neuron>>> layers;
    std::mt19937_64 rng;
    string loading[4] = {" | ", " / ", " - ", " \\ "};
    vector<vector<double>> conversionRates;
    shared_ptr<double> progressGoal = make_shared<double>(0.0);
    shared_ptr<double> curProgress = make_shared<double>(0.0);
    double learningRate;
    double momentum;
    double dropOutRate = 1.0;
    shared_ptr<bool> loadedData = make_shared<bool>(false);
    shared_ptr<bool> doneTraining = make_shared<bool>(false);
    bool conversions = false;
    bool verbose = false;
    shared_ptr<int> validationIters = make_shared<int>(0);
    int earlyStopping = -1;
    int barSize = 70;
};

