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

//neural network constructor, takes number of hidden layers + output layer, and neuron counts for each
NeuralNetwork::NeuralNetwork(vector<int> neurons, double learningRate, double momentum) {
    this->learningRate = learningRate;
    this->momentum = momentum;
    int numLayers = neurons.size();
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

NeuralNetwork::NeuralNetwork(string fileName) {
    loadData(fileName);
}

//min-max data normalization method
void NeuralNetwork::normalize(vector<vector<double>>& input, vector<double> minMaxRange, bool save, string fileName) {
    if (!conversions) {
        if (minMaxRange.size() > 0) {
            for (unsigned int p = 0; p < input[0].size(); p++) {
                vector<double> curData;
                for (auto &i: input) {
                    curData.push_back(i[p]);
                }
                double min = minMaxRange[0];
                double max = minMaxRange[1];
                for (auto &i: input) {
                    i[p] = (i[p] - min) / (max - min);
                    if (std::isnan(i[p])) {
                        i[p] = 0;
                    }
                }
                vector<double> tempFactors = {min, max};
                conversionRates.push_back(tempFactors);
            }
        }
        else {
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

//loads saved weight data
void NeuralNetwork::loadData(string fileName) {
    ifstream fin(fileName, ios::in);
    string numLayers, nPL, lr, m;
    //gets learning rate
    std::getline(fin, lr, ',');
    learningRate = stod(lr);
    //gets momentum
    std::getline(fin, m, '\n');
    momentum = stod(m);
    //gets number of layers
    std::getline(fin, numLayers, '\n');
    vector<int> neuronsPerLayer;
    //gets neuron counts for each layer
    for (unsigned int i = 0; i < stoi(numLayers) - 1; i++) {
        std::getline(fin, nPL, ',');
        neuronsPerLayer.push_back(stoi(nPL));
    }
    std::getline(fin, nPL, '\n');
    neuronsPerLayer.push_back(stoi(nPL));
    //for every neuron, gets weight count and weights
    vector<vector<Neuron*>> newLayers;
    for (int curLayer = 0; curLayer < neuronsPerLayer.size(); curLayer++) {
        vector<Neuron*> tempLayer;
        for (int curNeuron = 0; curNeuron < neuronsPerLayer[curLayer]; curNeuron++) {
            Neuron* tempNeuron = new Neuron;
            string numWeights;
            std::getline(fin, numWeights, '\n');
            vector<double> tempWeights;
            vector<double> tempGrads;
            string newWeight;
            for (int weightCount = 0; weightCount < stoi(numWeights) - 1; weightCount++) {

                std::getline(fin, newWeight, ',');
                tempWeights.push_back(stod(newWeight));
                tempGrads.push_back(0);
            }
            std::getline(fin, newWeight, '\n');
            tempWeights.push_back(stod(newWeight));
            tempGrads.push_back(0);
            tempNeuron->weights = tempWeights;
            tempNeuron->prevGradients = tempGrads;
            std::getline(fin, newWeight, '\n');
            tempNeuron->bias = stod(newWeight);
            tempNeuron->prevBias = 0;
            tempLayer.push_back(tempNeuron);
        }
        newLayers.push_back(tempLayer);
    }
    string numCov;
    std::getline(fin, numCov, '\n');
    vector<vector<double>> tempCovs;
    string newVal;
    for (int i = 0; i < stod(numCov); i++) {
        vector<double> newCov;
        std::getline(fin, newVal, ',');
        newCov.push_back(stod(newVal));
        std::getline(fin, newVal, '\n');
        newCov.push_back(stod(newVal));
        tempCovs.push_back(newCov);
    }
    conversionRates = tempCovs;
    layers = newLayers;
    loadedData = true;
    conversions = true;
}

//back propagation method, repeats for every iteration
void NeuralNetwork::train(vector<vector<double>> input, vector<vector<double>> allResults, int iterations, bool save, string fileName) {
    double lr = learningRate;
    double m = momentum;
    //initialize input neuron weights
    if (!loadedData) {
        for (unsigned int i = 0; i < layers[0].size(); i++) {
            initializeWeights(input[0].size(), layers[0][i], layers[1].size());
        }
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
                //current neuron
                auto curN = layers[layers.size()- 1][neuronCount];
                //gets the derivative of the neuron with respect to the expected output
                curN->delta = finalGradient(curN, desiredResult[neuronCount]);
                //adds the delta to the nextDeltas vector
                nextDeltas.push_back(curN->delta);
            }
            //hidden layer backprop for every hidden layer
            for (int layerCount = layers.size() - 2; layerCount >= 0; layerCount--) {
                //tempDeltas vector, will be the nextDeltas vector for the previous layer
                vector<double> tempDeltas;
                //for every neuron in the hidden layer
                for (unsigned int neuronCount = 0; neuronCount < layers[layerCount].size(); neuronCount++) {
                    //current enuron
                    auto curN = layers[layerCount][neuronCount];
                    //gets the derivative of the neuron with respect to the next layer neurons
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
                        //gets the derivative of weight adjust with the delta of the current neuron and the inputs
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
    loadedData = true;
    //save weight data
    if (save) {
        cout << "Saving data..." << endl;
        bool saveSuccess = saveData(fileName);
        if (saveSuccess) {
            cout << "Data saved successfully as " << fileName << endl;
        }
        else {
            cout << "Failed to save data" << endl;
        }
    }
}

void NeuralNetwork::trainMiniBatch(vector<vector<double>> input, vector<vector<double>> allResults, int iterations, int epochs, bool save, string fileName) {
    double lr = learningRate;
    double m = momentum;
    epochs += 1;
    //initialize input neuron weights
    if (!loadedData) {
        for (unsigned int i = 0; i < layers[0].size(); i++) {
            initializeWeights(input[0].size(), layers[0][i], layers[1].size());
        }
    }
    //for every iteration
    for (unsigned int z = 0; z < iterations; z++) {
        vector<vector<vector<double>>> batches;
        vector<vector<vector<double>>> batchResults;
        vector<int> indexes;
        indexes.reserve(input.size());
        for (int i = 0; i < input.size(); ++i)
            indexes.push_back(i);
        std::random_shuffle(indexes.begin(), indexes.end());

        int batchSize = floor(input.size() / epochs);
        //for every batch
        for (unsigned int i = 0; i < epochs - 1; i++) {
            vector<vector<double>> curBatch;
            vector<vector<double>> curResults;
            for (unsigned int x = i * batchSize; x < (i * batchSize) + batchSize; x++) {
                curBatch.push_back(input[indexes[i]]);
                curResults.push_back(allResults[indexes[i]]);
            }
            batches.push_back(curBatch);
            batchResults.push_back(curResults);
        }
        //for every batch
        for (unsigned int batchCount = 0; batchCount < batches.size(); batchCount++) {
            //reset neuron deltas
            for (int la = 0; la < layers.size(); la++) {
                for (int na = 0; na < layers[la].size(); na++) {
                    layers[la][na]->delta = 0;
                }
            }
            //for every training data point
            for (unsigned int x = 0; x < batches[batchCount].size(); x++) {
                //gets the actual result of the current data point
                auto desiredResult = batchResults[batchCount][x];
                //gets predicted result from forward propagation
                vector<double> finalResult = forwardProp(batches[batchCount][x]);
                //sets up the nextDelta variables for the hidden layers
                vector<double> nextDeltas;
                //output layer back propagation
                for (unsigned int neuronCount = 0; neuronCount < layers[layers.size() - 1].size(); neuronCount++) {
                    //current neuron
                    auto curN = layers[layers.size() - 1][neuronCount];
                    //gets the derivative of the neuron with respect to the expected output
                    curN->delta += finalGradient(curN, desiredResult[neuronCount]);
                    //adds the delta to the nextDeltas vector
                    nextDeltas.push_back(curN->delta);
                }
                //hidden layer backprop for every hidden layer
                for (int layerCount = layers.size() - 2; layerCount >= 0; layerCount--) {
                    //tempDeltas vector, will be the nextDeltas vector for the previous layer
                    vector<double> tempDeltas;
                    //for every neuron in the hidden layer
                    for (unsigned int neuronCount = 0; neuronCount < layers[layerCount].size(); neuronCount++) {
                        //current enuron
                        auto curN = layers[layerCount][neuronCount];
                        //gets the derivative of the neuron with respect to the next layer neurons
                        curN->delta += hiddenGradient(curN, neuronCount, layers[layerCount + 1], nextDeltas);
                        tempDeltas.push_back(curN->delta);
                    }
                    nextDeltas = tempDeltas;
                }
            }
            //updating weights in every layer
            for (int layerCount = layers.size() - 1; layerCount >= 0; layerCount--) {
                //for every neuron in the layer
                for (unsigned int neuronCount = 0; neuronCount < layers[layerCount].size(); neuronCount++) {
                    auto curN = layers[layerCount][neuronCount];
                    curN->delta /= batchSize;
                    //updates every weight and previous gradient for the current neuron
                    for (int w = 0; w < curN->weights.size(); w++) {
                        //gets the derivative of weight adjust with the delta of the current neuron and the inputs
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
        if (layers[layers.size() - 1].size() == 1) {
            newLayerResults.push_back(neuronResult);
        }
        else {
            //adds sigmoid activation of neuron result to layer results vector
            newLayerResults.push_back(sigmoid(neuronResult));
        }

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
    if (layers[layers.size() - 1].size() > 1) {
        double accuracy = 0;
        //for every test data point
        for (unsigned int i = 0; i < testData.size(); i++) {
            //gets forward propagation result with current test data point
            auto newResults = predictTest(testData[i]);
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
    else {
        double totalError = 0;
        for (unsigned int i = 0; i < testData.size(); i++) {
            auto result = predictTest(testData[i]);
            totalError += pow((result[0] - testLabel[i][0]), 2);
        }
        return totalError / testData.size();
    }
}

//Method for predicting a vector representing an unknown data point
vector<double> NeuralNetwork::predict(vector<double> unknownP) {
    vector<vector<double>> reformatUnknown = {unknownP};
    normalize(reformatUnknown);
    auto forwardResult = forwardProp(reformatUnknown[0]);
    if (layers[layers.size() - 1].size() == 1) {
        return forwardResult;
    }
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

void NeuralNetwork::resetWeights(int dataCount) {
    for (int lc = 0; lc < layers.size(); lc++) {
        //normalized xavier weight initialization
        int numOut = layers[lc].size();
        int numWeights;
        if (lc == 0) {
            numWeights = dataCount;
        }
        else {
            numWeights = layers[lc - 1].size();
        }
        std::uniform_real_distribution<double> unif(-1 * sqrt(6.0) / sqrt(numWeights + numOut), sqrt(6.0) / sqrt(numWeights + numOut));
        for (int n = 0; n < layers[lc].size(); n++) {
            auto newN = layers[lc][n];
            numWeights = newN->weights.size();
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
                newN->weights[i] = value;
                newN->prevGradients[i] = 0;
            }
            newN->prevBias = 0;
            newN->delta = 0;
        }
    }
}

//Private Methods
vector<double> NeuralNetwork::predictTest(vector<double> unknownP) {
    auto forwardResult = forwardProp(unknownP);
    if (layers[layers.size() - 1].size() == 1) {
        return forwardResult;
    }
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
    newN->delta = 0;
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

//saves weight data to csv file called "nn_save_config.csv"
bool NeuralNetwork::saveData(string fileName) {
    ofstream saveFile(fileName);
    if (saveFile) {
        //writes learning rate and momentum
        saveFile << learningRate << "," << momentum << "\n";
        //writes number of layers to the top
        saveFile << layers.size() << "\n";
        //writes neuron counts for each layer to the top
        for (unsigned int lCount = 0; lCount < layers.size() - 1; lCount++) {
            saveFile << layers[lCount].size() << ",";
        }
        saveFile << layers[layers.size() - 1].size() << "\n";
        for (unsigned int lCount = 0; lCount < layers.size(); lCount++) {
            //writes number of weights and weights underneath
            for (unsigned int nCount = 0; nCount < layers[lCount].size(); nCount++) {
                auto curN = layers[lCount][nCount];
                saveFile << curN->weights.size() << "\n";
                for (unsigned int wCount = 0; wCount < curN->weights.size() - 1; wCount++) {
                    saveFile << curN->weights[wCount] << ",";
                }
                saveFile << curN->weights[curN->weights.size() - 1] << "\n";
                saveFile << curN->bias << "\n";
            }
        }

        int conversionSize = conversionRates.size();
        saveFile << conversionSize << "\n";
        for (int i = 0; i < conversionSize - 1; i++) {
            saveFile << conversionRates[i][0] << "," << conversionRates[i][1] << "\n";
        }
        saveFile << conversionRates[conversionSize - 1][0] << "," << conversionRates[conversionSize - 1][1];
        saveFile.close();

        return true;
    }
    return false;
}


