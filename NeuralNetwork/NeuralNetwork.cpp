//
// Created by Alex Gilsoul on 10/19/2021.
//

#include "NeuralNetwork.h"


//Public Methods

double NeuralNetwork::Neuron::calculate(vector<double> input) {
    prevInputs = input;
    double total = 0;

    for (int i = 0; i < weights.size(); i++) {
        total += weights[i] * input[i];
    }
    total += bias;
    return total;
}

NeuralNetwork::NeuralNetwork(int numLayers, vector<int> neurons, double learningRate) {
    this->learningRate = learningRate;
    for (int i = 0; i < numLayers; i++) {
        vector<Neuron*> tempLayer;
        for (int n = 0; n < neurons[i]; n++) {
            Neuron* tempNeuron = new Neuron;
            if (i != 0) {
                initializeWeights(neurons[i-1], tempNeuron);
            }
            tempLayer.push_back(tempNeuron);
        }
        layers.push_back(tempLayer);
    }

    uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed>>32)};


    rng.seed(ss);
    //rng.seed(5);
}

//still need to code this and backpropogation method
void NeuralNetwork::train(vector<vector<double>> input, vector<double> desiredResult, int iterations) {
    //initialize input neuron weights
    for (int i = 0; i < layers[0].size(); i++) {
        initializeWeights(input.size(), layers[0][i]);
    }

    //for every iteration
    for (int i = 0; i < iterations; i++) {
        //for every input
        for (int x = 0; x < input.size(); x++) {
            //gets final result of forward propogation
            vector<double> finalResult = forwardProp(input[x]);
            //first layer backprop
            vector<double> nextDeltas;
            for (int neuronCount = 0; neuronCount < layers[layers.size() - 1].size();neuronCount++) {
                auto curN = layers[layers.size()- 1][neuronCount];
                curN->delta = finalGradient(curN, desiredResult[neuronCount]);
                nextDeltas.push_back(curN->delta);
            }
            //hidden layer backprop
            for (int layerCount = layers.size() - 2; layerCount >= 0; layerCount--) {
                //for every neuron in the layer
                vector<double> tempDeltas;
                for (int neuronCount = 0; neuronCount < layers[layerCount].size(); neuronCount++) {
                    auto curN = layers[layerCount][neuronCount];
                    curN->delta = hiddenGradient(curN, nextDeltas);
                    tempDeltas.push_back(curN->delta);
                    //for every weight
                    for (int w = 0; w < curN->weights.size(); w++) {
                        curN->weights[w] -= weightDerivative(curN->delta, curN->prevInputs[w]) * learningRate;
                    }
                    curN->bias -= curN->delta * learningRate;
                }
                nextDeltas = tempDeltas;
            }
        }
    }

}

double NeuralNetwork::sigmoid(double input) {
    return 1 / (1 + exp(-input));
}

double NeuralNetwork::sigmoidDeriv(double input) {
    return sigmoid(input) * (1 - sigmoid(input));
}


vector<double> NeuralNetwork::forwardProp(vector<double> input) {
    auto data = input;
    for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
        vector<double> layerResults;
        for (int neuronIndex = 0; neuronIndex < layers[layerIndex].size(); neuronIndex++) {
            auto tempNPointer = layers[layerIndex][neuronIndex];
            double neuronResult = tempNPointer->calculate(data);
            tempNPointer->prevInputs = data;
            tempNPointer->output = neuronResult;
            layerResults.push_back(sigmoid(neuronResult));
        }
        data = layerResults;
    }
    return data;
}


void NeuralNetwork::printVector(vector<double> input) {
    cout << "{ " << input[0];
    for (int i = 1; i < input.size(); i++) {
        cout << ", " << input[i];
    }
    cout << " }";
}





//Private Methods
void NeuralNetwork::initializeWeights(int numWeights, Neuron* newN) {
    std::uniform_real_distribution<double> unif(-1, 1);
    for (int i = 0; i < numWeights; i++) {
        newN->weights.push_back(unif(rng));
    }
}

//FINISH THESE
double NeuralNetwork::finalGradient(Neuron* curN, double expected) {
    return sigmoidDeriv(curN->output) * (sigmoid(curN->output) - expected);
}

double NeuralNetwork::hiddenGradient(Neuron* curN, vector<double> nextDeltas) {
    double total = 0;
    for (int i = 0; i < curN->weights.size(); i++) {
        total += curN->weights[i] * nextDeltas[i];
    }
    return sigmoidDeriv(curN->output) * total;
}

double NeuralNetwork::weightDerivative(double neuronError, double prevNeuron) {
    return neuronError * prevNeuron;
}


//Cost function partial derivative with respect to the weights
double NeuralNetwork::derivWeight(Neuron* curN, int index, double expected) {
    double prediction = curN->calculate(curN->prevInputs);
    return 2 * (expected - (prediction)) * -curN->prevInputs[index];
}

//Cost function partial derivative with respect to the bias
double NeuralNetwork::derivBias(Neuron* curN, double expected) {
    double prediction = curN->calculate(curN->prevInputs);
    return -2 * (expected - prediction);
}


