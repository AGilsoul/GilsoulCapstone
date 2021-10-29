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
void NeuralNetwork::train(vector<vector<double>> input, vector<vector<double>> allResults, int iterations) {
    //initialize input neuron weights
    for (int i = 0; i < layers[0].size(); i++) {
        initializeWeights(input.size(), layers[0][i]);
    }
    //for every iteration
    for (int i = 0; i < iterations; i++) {
        cout << i << endl;
        for (int x = 0; x < input.size(); x++) {
            printVector(forwardProp(input[x]));
            cout << endl;
            auto desiredResult = allResults[x];
            cout << "data point #" << x << endl;
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
                    curN->delta = hiddenGradient(curN, neuronCount, layers[layerCount + 1], nextDeltas);
                    tempDeltas.push_back(curN->delta);
                    //for every weight
                }
                nextDeltas = tempDeltas;
            }

            //updating weights
            for (int layerCount = layers.size() - 1; layerCount >= 0; layerCount--) {
                for (int neuronCount = 0; neuronCount < layers[layerCount].size(); neuronCount++) {
                    auto curN = layers[layerCount][neuronCount];
                    for (int w = 0; w < curN->weights.size(); w++) {
                        curN->weights[w] -= weightDerivative(curN->delta, curN->prevInputs[w]) * learningRate;
                    }
                    curN->bias -= curN->delta * learningRate;
                }
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


vector<vector<double>> NeuralNetwork::vectorSplit(vector<vector<double>> vec, int start, int fin) {
    vector<vector<double>> newVec;
    for (int i = start; i <= fin; i++) {
        newVec.push_back(vec[i]);
    }
    return newVec;
}

double NeuralNetwork::test(vector<vector<double>> testData, vector<vector<double>> testLabel) {
    double accuracy = 0;
    for (int i = 0; i < testData.size(); i++) {
        vector<double> tempResult = forwardProp(testData[i]);
        cout << "Actual: ";
        printVector(tempResult);
        cout << endl << "Expected: ";
        printVector(testLabel[i]);
        vector<double> newResult;
        double max = 0;
        int maxIndex;
        for (int y = 0; y < tempResult.size(); y++) {
            if (tempResult[y] > max) {
                maxIndex = y;
                max = tempResult[y];
            }
        }
        for (int z = 0; z < tempResult.size(); z++) {
            if (z == maxIndex) {
                newResult.push_back(1);
            }
            else {
                newResult.push_back(0);
            }
        }
        bool correct = true;
        for (int x = 0; x < testLabel[i].size(); x++) {
            if (tempResult[x] != testLabel[i][x]) {
                correct = false;
            }
        }
        if (correct) {
            accuracy++;
        }
    }
    return accuracy / testData.size() * 100;
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

double NeuralNetwork::hiddenGradient(Neuron* curN, int nIndex, vector<Neuron*> nextLayer, vector<double> nextDeltas) {
    double total = 0;
    for (int i = 0; i < nextLayer.size(); i++) {
        auto newN = nextLayer[i];
        total += newN->weights[nIndex] * nextDeltas[i];
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


