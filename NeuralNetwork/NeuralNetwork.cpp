//
// Created by Alex Gilsoul on 10/19/2021.
//

#include "NeuralNetwork.h"


//Public Methods
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

    uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed>>32)};
    rng.seed(ss);
}

//still need to code this and backpropogation method
void NeuralNetwork::train(vector<vector<double>> input, vector<double> desiredResult) {
    for (int i = 0; i < layers[0].size(); i++) {
        initializeWeights(input.size(), layers[0][i]);
    }
    for (int i = 0; i < layers.size(); i++) {
        for (int x = 0; x < layers[i].size(); x++) {
            printVector(layers[i][x].weights);
            cout << " ";
        }
        cout << endl;
    }
    /*
    for (int x = 0; x < input.size(); x++) {
        vector<double> finalResult = forwardProp(input[x]);
        vector<double> prevNeuronChange;
        //STOPPED HERE
    }
     */

}

double NeuralNetwork::sigmoid(double input) {
    return 1 / (1 + exp(-input));
}

double NeuralNetwork::sigmoidDeriv(double input) {
    return sigmoid(input) * (1 - sigmoid(input));
}

void NeuralNetwork::printVector(vector<double> input) {
    cout << "{ " << input[0];
    for (int i = 1; i < input.size(); i++) {
        cout << ", " << input[i];
    }
    cout << " }";
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


//Private Methods
void NeuralNetwork::initializeWeights(int numWeights, Neuron& newN) {
    std::uniform_real_distribution<double> unif(-1, 1);
    for (int i = 0; i < numWeights; i++) {
        newN.weights.push_back(unif(rng));
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

void NeuralNetwork::backProp() {

}

/*
 * Not sure if all the following methods are necessary for the network
 * I included them just in case, might delete later
 */
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


