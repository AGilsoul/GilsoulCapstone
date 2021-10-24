//
// Created by agils on 10/19/2021.
//

#include "NeuralNetwork.h"
#include <cmath>



NeuralNetwork::NeuralNetwork(int numLayers, vector<int> neurons, double learningRate) {
    this->learningRate = learningRate;
    for (int i = 0; i < numLayers; i++) {
        vector<Neuron> tempLayer;
        for (int n = 0; n < neurons[i]; n++) {
            Neuron tempNeuron;
            if (i != 0) {
                initializeWeights(neurons[i-1], tempNeuron);
            }
            tempLayer.push_back(tempNeuron);
        }
        layers.push_back(tempLayer);
    }
}

//not yet working, will train the neural network by 
void NeuralNetwork::train(vector<vector<double>> input, vector<double> desiredResult) {
    for (int i = 0; i < layers[0].size(); i++) {
        initializeWeights(input.size(), layers[0][i]);
    }
    for (int x = 0; x < input.size(); x++) {
        vector<double> finalResult = forwardProp(input[x]);
        vector<double> prevNeuronChange;
        //STOPPED HERE
    }

}


vector<double> NeuralNetwork::forwardProp(vector<double> input) {
    auto data = input;
    for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
        vector<double> layerResults;
        for (int neuronIndex = 0; neuronIndex < layers[layerIndex].size(); neuronIndex++) {
            double neuronResults = layers[layerIndex][neuronIndex].calculate(data);
            layerResults.push_back(neuronResults);
        }
        data = layerResults;
    }
    return data;
}

double NeuralNetwork::sigmoid(double input) {
    return 1 / (1 + exp(-input));
}


double NeuralNetwork::Neuron::calculate(vector<double> input) {
    prevInputs = input;
    double total = 0;

    for (int i = 0; i < weights.size(); i++) {
        total += weights[i] * input[i];
    }
    total += bias;
    return sigmoid(total);
}


void NeuralNetwork::Neuron::calcWeightChange(double desired) {
    for (int m = 0; m < weights.size(); m++) {
        avgWeightChange[m].push_back(derivWeight(*this, desired, m));
    }
    avgBiasChange.push_back(derivBias(*this, desired));
}


void NeuralNetwork::initializeWeights(int numWeights, Neuron& newN) {
    for (int i = 0; i < numWeights; i++) {
        newN.weights.push_back(0);
        vector<double> tempWeightChange;
        newN.avgWeightChange.push_back(tempWeightChange);
    }
}


double NeuralNetwork::costFunction(NeuralNetwork::Neuron input, double desired) {
    double total = 0;
    for (int i = 0; i < input.weights.size(); i++) {
        int result = pow(input.calculate(input.prevInputs) - desired, 2);
        total += result;
    }
    return (total / input.weights.size());
}


double NeuralNetwork::derivWeight(NeuralNetwork::Neuron input, double desired, int index) {
    double total = 0;
    for (int i = 0; i < input.weights.size(); i++) {
        double result = 2 * (desired - (input.calculate(input.prevInputs))) * -input.prevInputs[index];
    }
    return total / input.weights.size();
}


double NeuralNetwork::derivBias(NeuralNetwork::Neuron input, double desired) {
    double total = 0;
    for (int i = 0; i < input.weights.size(); i++) {
        double prediction = input.calculate(input.prevInputs);
        total += -2 * (desired - prediction);
    }
    return total / input.weights.size();
}


double NeuralNetwork::derivInput(NeuralNetwork::Neuron input, double desired) {
    double total = 0;
    for (int i = 0; i < input.weights.size(); i++) {
        double result = 2 * (desired - (input.calculate(input.prevInputs))) * -input.weights[i];
        total += result;
    }
    return total / input.weights.size();
}


void NeuralNetwork::printVector(vector<double> input) {
    std::cout << "{ " << input[0];
    for (int i = 1; i < input.size(); i++) {
        std::cout << ", " << input[i];
    }
    std::cout << " }" << std::endl;
}