//
// Created by Alex Gilsoul on 10/19/2021.
//

#include "NeuralNetwork.h"


//Public Methods

//calculation method for a single neuron
double NeuralNetwork::Neuron::calculate(vector<double> input) {
    prevInputs = input;
    double total = 0;
    for (unsigned int i = 0; i < weights.size(); i++) {
        total += weights[i] * input[i];
    }
    total += bias;
    return total;
}

//neural network constructor
NeuralNetwork::NeuralNetwork(int numLayers, vector<int> neurons, double learningRate, double momentum) {
    this->learningRate = learningRate;
    this->momentum = momentum;
    uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed>>32)};
    rng.seed(ss);
    //rng.seed(5);

    //initializes neurons and weights for every layer except input layer
    for (unsigned int i = 0; i < numLayers; i++) {
        vector<Neuron*> tempLayer;
        for (unsigned int n = 0; n < neurons[i]; n++) {
            Neuron* tempNeuron = new Neuron;
            if (i != 0) {
                initializeWeights(neurons[i-1], tempNeuron, neurons[i]);
            }
            tempLayer.push_back(tempNeuron);
        }
        layers.push_back(tempLayer);
    }
}

//min-max data normalization method
void NeuralNetwork::normalize(vector<vector<double>>& input) {
    if (!conversions) {
        for (unsigned int p = 0; p < input[0].size(); p++) {
            vector<double> curData;
            for (auto &i: input) {
                curData.push_back(i[p]);
            }
            auto sortedData = sortVector(curData);
            double min = sortedData[0];
            double max = sortedData[sortedData.size() - 1];
            for (auto &i: input) {
                i[p] = (i[p] - min) / (max - min);
                if (std::isnan(i[p])) {
                    i[p] = 0;
                }
            }
            vector<double> tempFactors = {min, max};
            conversionRates.push_back(tempFactors);
        }
        conversions = true;


    }
    else {
        for (unsigned int p = 0; p < input[0].size(); p++) {
            for (auto &i: input) {
                i[p] = (i[p] - conversionRates[p][0]) / (conversionRates[p][1] - conversionRates[p][0]);
                if (std::isnan(i[p])) {
                    i[p] = 0;
                }
            }
        }
    }
}

//back propagation method, repeats for every iteration
void NeuralNetwork::train(vector<vector<double>> input, vector<vector<double>> allResults, int iterations) {
    double lr = learningRate;
    double m = momentum;
    //initialize input neuron weights
    for (unsigned int i = 0; i < layers[0].size(); i++) {
        initializeWeights(input.size(), layers[0][i], layers[1].size());
    }
    //for every iteration
    for (unsigned int z = 0; z < iterations; z++) {
        //for every training data point
        for (unsigned int x = 0; x < input.size(); x++) {
            //gets the actual result of the current data point
            auto desiredResult = allResults[x];
            //gets predicted result from forward propagation
            vector<double> finalResult = forwardProp(input[x]);
            //sets up the nextDelta variables for the hidden layers
            vector<double> nextDeltas;
            //output layer back propagation
            for (unsigned int neuronCount = 0; neuronCount < layers[layers.size() - 1].size();neuronCount++) {
                auto curN = layers[layers.size()- 1][neuronCount];
                curN->delta = finalGradient(curN, desiredResult[neuronCount]);
                nextDeltas.push_back(curN->delta);
            }
            //hidden layer backprop for every hidden layer
            for (int layerCount = layers.size() - 2; layerCount >= 0; layerCount--) {
                //tempDeltas vector, will be the nextDeltas vector for the previous layer
                vector<double> tempDeltas;
                //for every neuron in the hidden layer
                for (unsigned int neuronCount = 0; neuronCount < layers[layerCount].size(); neuronCount++) {
                    auto curN = layers[layerCount][neuronCount];
                    curN->delta = hiddenGradient(curN, neuronCount, layers[layerCount + 1], nextDeltas);
                    tempDeltas.push_back(curN->delta);
                }
                nextDeltas = tempDeltas;
            }

            //updating weights in every layer
            for (int layerCount = layers.size() - 1; layerCount >= 0; layerCount--) {
                //for every neuron in the layer
                for (unsigned int neuronCount = 0; neuronCount < layers[layerCount].size(); neuronCount++) {
                    auto curN = layers[layerCount][neuronCount];
                    //updates every weight and previous gradient for the current neuron
                    for (int w = 0; w < curN->weights.size(); w++) {
                        double result = weightDerivative(curN->delta, curN->prevInputs[w]) * lr;
                        curN->weights[w] -= result + curN->prevGradients[w] * m;
                        curN->prevGradients[w] = result + curN->prevGradients[w] * m;
                    }
                    //updates bias and previous bias for the current neuron
                    double bResult = curN->delta * lr;
                    curN->bias -= bResult + curN->prevBias * m;
                    curN->prevBias = bResult + curN->prevBias * m;
                }
            }
        }
    }
}

//forward propagation method
vector<double> NeuralNetwork::forwardProp(vector<double> input) {
    auto data = input;
    //for every hidden layer
    for (int layerIndex = 0; layerIndex < layers.size() - 1; layerIndex++) {
        vector<double> layerResults;
        //for every neuron in each layer
        for (unsigned int neuronIndex = 0; neuronIndex < layers[layerIndex].size(); neuronIndex++) {
            auto tempNPointer = layers[layerIndex][neuronIndex];
            //calculates neuron output
            double neuronResult = tempNPointer->calculate(data);
            tempNPointer->prevInputs = data;
            tempNPointer->output = neuronResult;
            //adds ReLu activation of neuron calculation to layer results vector
            layerResults.push_back(relu(neuronResult));
        }
        //forward propagates the results to the next layer
        data = layerResults;
    }

    vector<double> newLayerResults;
    //for each neuron in the output layer
    for (unsigned int neuronIndex = 0; neuronIndex < layers[layers.size() - 1].size(); neuronIndex++) {
        auto tempNPointer = layers[layers.size() - 1][neuronIndex];
        //calculates current neuron output
        double neuronResult = tempNPointer->calculate(data);
        tempNPointer->prevInputs = data;
        tempNPointer->output = neuronResult;
        //adds sigmoid activation of neuron result to layer results vector
        newLayerResults.push_back(sigmoid(neuronResult));
    }
    data = newLayerResults;
    return data;
}

//static method for printing vectors of doubles
void NeuralNetwork::printVector(vector<double> input) {
    cout << "{ " << input[0];
    for (unsigned int i = 1; i < input.size(); i++) {
        cout << ", " << input[i];
    }
    cout << " }";
}

//Splits up a 2D vector of doubles based on specified start/end indices
vector<vector<double>> NeuralNetwork::vectorSplit(vector<vector<double>> vec, int start, int fin) {
    vector<vector<double>> newVec;
    for (unsigned int i = start; i <= fin; i++) {
        newVec.push_back(vec[i]);
    }
    return newVec;
}

//testing method, compares predicted results after training with actual results
double NeuralNetwork::test(vector<vector<double>>& testData, vector<vector<double>>& testLabel) {
    double accuracy = 0;
    //for every test data point
    for (unsigned int i = 0; i < testData.size(); i++) {
        //gets forward propagation result with current test data point
        vector<double> tempResult = forwardProp(testData[i]);
        int maxIndex;
        double maxVal = 0;
        //finds highest value index from result
        for (int x = 0; x < tempResult.size(); x++) {
            if (tempResult[x] > maxVal) {
                maxIndex = x;
                maxVal = tempResult[x];
            }
        }
        vector<double> newResults;
        //sets highest value index of result equal to 1 and the rest to 0
        for(int x = 0; x < tempResult.size(); x++) {
            if (x == maxIndex) {
                newResults.push_back(1);
            }
            else {
                newResults.push_back(0);
            }
        }
        bool correct = true;
        //compares predicted answer with actual answer
        for (int x = 0; x < testLabel[i].size(); x++) {
            if (newResults[x] != testLabel[i][x]) {
                correct = false;
            }
        }
        if (correct) { accuracy++; }

    }
    return accuracy / testData.size() * 100;
}

//Method for predicting a vector representing an unknown data point
vector<double> NeuralNetwork::predict(vector<double> unknownP) {
    vector<vector<double>> reformatUnknown = {unknownP};
    normalize(reformatUnknown);
    auto forwardResult = forwardProp(reformatUnknown[0]);
    vector<double> newResult;
    int maxIndex;
    double maxVal = 0;
    for (int i = 0; i < forwardResult.size(); i++) {
        if (forwardResult[i] > maxVal) {
            maxVal = forwardResult[i];
            maxIndex = i;
        }
    }
    for (int i = 0; i < forwardResult.size(); i++) {
        if (i == maxIndex) {
            newResult.push_back(1);
        }
        else {
            newResult.push_back(0);
        }
    }
    return newResult;
}


//Private Methods
//sigmoid activation function
double NeuralNetwork::sigmoid(double input) {
    return 1 / (1 + exp(-input));
}

//derivative of sigmoid function
double NeuralNetwork::sigmoidDeriv(double input) {
    return sigmoid(input) * (1 - sigmoid(input));
}

//ReLu activation function
double NeuralNetwork::relu(double input) {
    if (input > 0) {
        return input;
    }
    return 0;
}

//ReLu function derivative, slightly modified
double NeuralNetwork::reluDeriv(double input) {
    if (input > 0) {
        return 1;
    }
    return 0;
}

//initializes neuron weights to a random value
void NeuralNetwork::initializeWeights(int numWeights, Neuron* newN, double numOut) {
    //normalized xavier weight initialization
    std::uniform_real_distribution<double> unif(-1 * sqrt(6.0) / sqrt(numWeights + numOut), sqrt(6.0) / sqrt(numWeights + numOut));
    bool notZero;
    for (unsigned int i = 0; i < numWeights; i++) {
        notZero = false;
        double value;
        while (!notZero) {
            value = unif(rng);
            if (value != 0 && !std::isnan(value)) {
                notZero = true;
            }
        }
        newN->weights.push_back(value);
        newN->prevGradients.push_back(0);
    }
    newN->prevBias = 0;
}

//gradient descent method for final layer
double NeuralNetwork::finalGradient(Neuron* curN, double expected) {
    return sigmoidDeriv(curN->output) * (sigmoid(curN->output) - expected);
}

//gradient descent method for hidden layers
double NeuralNetwork::hiddenGradient(Neuron* curN, int nIndex, vector<Neuron*> nextLayer, vector<double> nextDeltas) {
    double total = 0;
    for (unsigned int i = 0; i < nextLayer.size(); i++) {
        auto newN = nextLayer[i];
        total += newN->weights[nIndex] * nextDeltas[i];
    }
    return reluDeriv(curN->output) * total;
}

//Gets the derivative to be applied to weights
double NeuralNetwork::weightDerivative(double neuronError, double prevNeuron) {
    return neuronError * prevNeuron;
}

//sorts a vector of doubles
vector<double> NeuralNetwork::sortVector(vector<double> vec) {
    vector<double> sortedData;
    sortedData.push_back(vec[0]);
    for (unsigned int x = 1; x < vec.size(); x++) {
        for (unsigned int y = 0; y < sortedData.size(); y++) {
            if (vec[x] < sortedData[y]) {
                sortedData.insert(sortedData.begin() + y, vec[x]);
                break;
            }
            else if (y == sortedData.size() - 1) {
                sortedData.push_back(vec[x]);
                break;
            }
        }
    }
    return sortedData;
}


