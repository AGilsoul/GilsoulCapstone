//
// Created by Alex Gilsoul on 10/19/2021.
//

#include "../include/NeuralNetwork.h"


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
NeuralNetwork::NeuralNetwork(vector<int> neurons, double learningRate, double momentum, bool verbose, int barSize) {
    this->learningRate = learningRate;
    this->momentum = momentum;
    this->verbose = verbose;
    this->barSize = barSize;
    int numLayers = neurons.size();
    uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed>>32)};
    rng.seed(ss);

    //initializes neurons and weights for every layer except input layer
    for (unsigned int i = 0; i < numLayers; i++) {
        vector<shared_ptr<Neuron>> tempLayer;
        for (unsigned int n = 0; n < neurons[i]; n++) {
            auto tempNeuron = make_shared<Neuron>();
            if (i != 0) {
                initializeWeights(neurons[i-1], tempNeuron, neurons[i]);
            }
            tempLayer.push_back(tempNeuron);
        }
        layers.push_back(tempLayer);
    }
}

NeuralNetwork::NeuralNetwork(string fileName, bool verbose, int barSize) {
    loadData(fileName);
    this->verbose = verbose;
    this->barSize = barSize;
    *loadedData = true;
}

//min-max data normalization method
void NeuralNetwork::normalize(vector<vector<double>>& input, vector<double> minMaxRange) {
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
    vector<vector<shared_ptr<Neuron>>> newLayers;
    for (int curLayer = 0; curLayer < neuronsPerLayer.size(); curLayer++) {
        vector<shared_ptr<Neuron>> tempLayer;
        for (int curNeuron = 0; curNeuron < neuronsPerLayer[curLayer]; curNeuron++) {
            auto tempNeuron = make_shared<Neuron>();
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
    *loadedData = true;
    conversions = true;
}

//back propagation method, repeats for every iteration
void NeuralNetwork::trainWithValidation(vector<vector<double>> trainInput, vector<vector<double>> trainResults,vector<vector<double>> valInput, vector<vector<double>> valResults, int minIterations, int maxIterations) {
    *curProgress = 0;
    *progressGoal = 1;
    *doneTraining = false;
    //initialize input neuron weights
    if (!*loadedData) {
        for (unsigned int i = 0; i < layers[0].size(); i++) {
            initializeWeights(trainInput[0].size(), layers[0][i], layers[1].size());
        }
    }
    else {
        resetGradients();
    }
    if (this->verbose) cout << "SGD Progress:" << endl;
    *timeDuration = 0.0;
    curWatch->reset();
    std::thread thread_obj(&NeuralNetwork::_trainWithValidation, *this, trainInput, trainResults, valInput, valResults, minIterations, maxIterations);
    if (verbose) {
        std::thread thread_progress(&NeuralNetwork::progressBar, *this);
        thread_progress.join();
    }
    thread_obj.join();
    cout << "Trained for " << *validationIters << " iterations" << endl;
}

void NeuralNetwork::_trainWithValidation(vector<vector<double>> trainInput, vector<vector<double>> trainResults, vector<vector<double>> valInput, vector<vector<double>> valResults, int minIterations, int maxIterations) {
    double lr = learningRate;
    double m = momentum;
    *progressGoal = maxIterations * trainInput.size();
    double prevAccuracy = 0.0;
    int iterationsDecreased = 0;
    //for every iteration
    for (unsigned int z = 0; z < maxIterations; z++) {
        *validationIters += 1;
        //for every training data point
        for (unsigned int x = 0; x < trainInput.size(); x++) {
            curWatch->reset();
            *curProgress += 1;
            //gets the actual result of the current data point
            auto desiredResult = trainResults[x];
            //gets predicted result from forward propagation
            vector<double> finalResult = forwardProp(trainInput[x], this->dropOutRate);
            //sets up the nextDelta variables for the hidden layers
            vector<double> nextDeltas;
            //output layer back propagation
            if (layers[layers.size()-1].size() == 1) {
                for (unsigned int neuronCount = 0; neuronCount < layers[layers.size() - 1].size();neuronCount++) {
                    //current neuron
                    auto curN = layers[layers.size()- 1][neuronCount];
                    //gets the derivative of the neuron with respect to the expected output
                    curN->delta = finalLinearGradient(curN, desiredResult[neuronCount]);

                    //adds the delta to the nextDeltas vector
                    nextDeltas.push_back(curN->delta);
                }
            }
            else {
                nextDeltas = finalSoftmaxGradient(desiredResult);
                for (int i = 0; i < layers[layers.size()-1].size(); i++) {
                    auto curN = layers[layers.size()- 1][i];
                    curN->delta = nextDeltas[i];
                }
            }


            //hidden layer backprop for every hidden layer
            for (int layerCount = layers.size() - 2; layerCount >= 0; layerCount--) {
                //tempDeltas vector, will be the nextDeltas vector for the previous layer
                vector<double> tempDeltas;
                //for every neuron in the hidden layer
                for (unsigned int neuronCount = 0; neuronCount < layers[layerCount].size(); neuronCount++) {
                    //current neuron
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
                        double result = weightDerivative(curN->weights[w], curN->delta, curN->prevInputs[w]) * lr;
                        curN->weights[w] -= result + curN->prevGradients[w] * m;
                        curN->prevGradients[w] = result + curN->prevGradients[w] * m;
                    }
                    //updates bias and previous bias for the current neuron
                    double bResult = curN->delta * lr;
                    curN->bias -= bResult + curN->prevBias * m;
                    curN->prevBias = bResult + curN->prevBias * m;
                }
            }
            auto tempDuration = curWatch->elapsed_time() * 60 * (*progressGoal - *curProgress);
            if (fabs(tempDuration - *timeDuration) / *timeDuration >= 0.075) {
                *timeDuration = tempDuration;
            }
        }
        double valAccuracy = test(valInput, valResults);
        double trainAccuracy = test(trainInput, trainResults);
        if (valAccuracy < prevAccuracy and z >= minIterations) {
            iterationsDecreased++;
        }
        else {
            iterationsDecreased = 0;
            prevAccuracy = valAccuracy;
        }
        if (iterationsDecreased >= this->earlyStopping) {
            break;
        }
    }
    *doneTraining = true;
    *loadedData = true;
}

void NeuralNetwork::train(vector<vector<double>> input, vector<vector<double>> allResults, int iterations) {
    *curProgress = 0;
    *progressGoal = 1;
    *doneTraining = false;
    // initialize input neuron weights
    if (!*loadedData) {
        for (unsigned int i = 0; i < layers[0].size(); i++) {
            initializeWeights(input[0].size(), layers[0][i], layers[1].size());
        }
    }
    else {
        resetGradients();
    }
    if (this->verbose) cout << "SGD Progress:" << endl;
    *timeDuration = 0.0;
    curWatch->reset();
    std::thread thread_obj(&NeuralNetwork::_train, *this, input, allResults, iterations);
    if (verbose) {
        std::thread thread_progress(&NeuralNetwork::progressBar, *this);
        thread_progress.join();
    }
    thread_obj.join();
}

void NeuralNetwork::_train(vector<vector<double>> input, vector<vector<double>> allResults, int iterations) {
    double lr = learningRate;
    double m = momentum;
    *progressGoal = iterations * input.size();
    //for every iteration
    for (unsigned int z = 0; z < iterations; z++) {
        //for every training data point
        for (unsigned int x = 0; x < input.size(); x++) {
            curWatch->reset();
            *curProgress += 1;
            //gets the actual result of the current data point
            auto desiredResult = allResults[x];
            //gets predicted result from forward propagation
            vector<double> finalResult = forwardProp(input[x], this->dropOutRate);
            //sets up the nextDelta variables for the hidden layers
            vector<double> nextDeltas;
            //output layer back propagation
            if (layers[layers.size()-1].size() == 1) {
                for (unsigned int neuronCount = 0; neuronCount < layers[layers.size() - 1].size();neuronCount++) {
                    //current neuron
                    auto curN = layers[layers.size()- 1][neuronCount];
                    //gets the derivative of the neuron with respect to the expected output
                    curN->delta = finalLinearGradient(curN, desiredResult[neuronCount]);
                    //adds the delta to the nextDeltas vector
                    nextDeltas.push_back(curN->delta);
                }
            }
            else {
                nextDeltas = finalSoftmaxGradient(desiredResult);
                for (int i = 0; i < layers[layers.size()-1].size(); i++) {
                    auto curN = layers[layers.size()- 1][i];
                    curN->delta = nextDeltas[i];
                }
            }

            //hidden layer backprop for every hidden layer
            for (int layerCount = layers.size() - 2; layerCount >= 0; layerCount--) {
                //tempDeltas vector, will be the nextDeltas vector for the previous layer
                vector<double> tempDeltas;
                //for every neuron in the hidden layer
                for (unsigned int neuronCount = 0; neuronCount < layers[layerCount].size(); neuronCount++) {
                    //current neuron
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
                        double result = weightDerivative(curN->weights[w], curN->delta, curN->prevInputs[w]) * lr;
                        curN->weights[w] -= result + curN->prevGradients[w] * m;
                        curN->prevGradients[w] = result + curN->prevGradients[w] * m;
                    }
                    //updates bias and previous bias for the current neuron
                    double bResult = curN->delta * lr;
                    curN->bias -= bResult + curN->prevBias * m;
                    curN->prevBias = bResult + curN->prevBias * m;
                }
            }
            auto tempDuration = curWatch->elapsed_time() * 60 * (*progressGoal - *curProgress);
            if ((fabs(tempDuration - *timeDuration) / *timeDuration) >= 0.075) {
                *timeDuration = tempDuration;
            }
        }
    }
    *doneTraining = true;
    *loadedData = true;
}

void NeuralNetwork::trainMiniBatch(vector<vector<double>> input, vector<vector<double>> allResults, int iterations, int batchSize) {
    *curProgress = 0;
    *progressGoal = 1;
    *doneTraining = false;
    // initialize input neuron weights
    if (!*loadedData) {
        for (unsigned int i = 0; i < layers[0].size(); i++) {
            initializeWeights(input[0].size(), layers[0][i], layers[1].size());
        }
    }
    else {
        resetGradients();
    }
    if (this->verbose) cout << "Mini-Batch Progress:" << endl;
    curWatch->reset();
    std::thread thread_obj(&NeuralNetwork::_trainMiniBatch, *this, input, allResults, iterations, batchSize);
    if (verbose) {
        std::thread thread_progress(&NeuralNetwork::progressBar, *this);
        thread_progress.join();
    }
    thread_obj.join();
}

void NeuralNetwork::_trainMiniBatch(vector<vector<double>> input, vector<vector<double>> allResults, int iterations, int batchSize) {
    double lr = learningRate;
    double m = momentum;
    *progressGoal = iterations * input.size();
    //for every iteration
    for (unsigned int z = 0; z < iterations; z++) {
        vector<vector<vector<double>>> batches;
        vector<vector<vector<double>>> batchResults;

        int numBatches = floor(input.size() / batchSize);

        vector<int> indexes;
        indexes.reserve(input.size());
        for (int i = 0; i < input.size(); ++i)
            indexes.push_back(i);

        std::random_shuffle(indexes.begin(), indexes.end());

        //for every batch
        for (unsigned int i = 0; i < numBatches; i++) {
            vector<vector<double>> curBatch;
            vector<vector<double>> curResults;
            for (unsigned int x = i * batchSize; x < (i * batchSize) + batchSize; x++) {
                curBatch.push_back(input[indexes[x]]);
                curResults.push_back(allResults[indexes[x]]);
            }
            batches.push_back(curBatch);
            batchResults.push_back(curResults);
        }


        //for every batch
        int timeDurations = 0;
        for (unsigned int batchCount = 0; batchCount < batches.size(); batchCount++) {
            //reset neuron deltas
            for (int la = 0; la < layers.size(); la++) {
                for (int na = 0; na < layers[la].size(); na++) {
                    layers[la][na]->delta = 0;
                }
            }
            curWatch->reset();
            //for every training data point
            for (unsigned int x = 0; x < batches[batchCount].size(); x++) {
                *curProgress += 1;
                //gets the actual result of the current data point
                auto desiredResult = batchResults[batchCount][x];
                //gets predicted result from forward propagation
                vector<double> finalResult = forwardProp(batches[batchCount][x], this->dropOutRate);
                //sets up the nextDelta variables for the hidden layers
                vector<double> nextDeltas;
                //output layer back propagation
                if (layers[layers.size()-1].size() == 1) {
                    for (unsigned int neuronCount = 0; neuronCount < layers[layers.size() - 1].size();neuronCount++) {
                        //current neuron
                        auto curN = layers[layers.size()- 1][neuronCount];
                        //gets the derivative of the neuron with respect to the expected output
                        curN->delta += finalLinearGradient(curN, desiredResult[neuronCount]);

                        //adds the delta to the nextDeltas vector
                        nextDeltas.push_back(curN->delta);
                    }
                }
                else {
                    nextDeltas = finalSoftmaxGradient(desiredResult);
                    for (int i = 0; i < layers[layers.size()-1].size(); i++) {
                        auto curN = layers[layers.size()- 1][i];
                        curN->delta += nextDeltas[i];
                    }
                }

                //hidden layer backprop for every hidden layer
                for (int layerCount = layers.size() - 2; layerCount >= 0; layerCount--) {
                    //tempDeltas vector, will be the nextDeltas vector for the previous layer
                    vector<double> tempDeltas;
                    //for every neuron in the hidden layer
                    for (unsigned int neuronCount = 0; neuronCount < layers[layerCount].size(); neuronCount++) {
                        //current neuron
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
                    curN->delta /= batches[batchCount].size();
                    //updates every weight and previous gradient for the current neuron
                    for (int w = 0; w < curN->weights.size(); w++) {
                        //gets the derivative of weight adjust with the delta of the current neuron and the inputs
                        double result = weightDerivative(curN->weights[w], curN->delta, curN->prevInputs[w]) * lr;
                        curN->weights[w] -= (result + curN->prevGradients[w] * m);
                        curN->prevGradients[w] = (result + curN->prevGradients[w] * m);
                    }
                    //updates bias and previous bias for the current neuron
                    double bResult = curN->delta * lr;
                    curN->bias -= bResult + curN->prevBias * m;
                    curN->prevBias = bResult + curN->prevBias * m;
                }
            }
            auto tempDuration = (curWatch->elapsed_time()) * (*progressGoal - *curProgress);
            if (fabs(tempDuration - *timeDuration) / *timeDuration >= 0.075) {
                *timeDuration = tempDuration;
            }

        }
    }
    *doneTraining = true;
    *loadedData = true;
}

//forward propagation method
vector<double> NeuralNetwork::forwardProp(vector<double> input, double chanceDropout) {
    auto data = input;
    //for every hidden layer
    for (int layerIndex = 0; layerIndex < layers.size() - 1; layerIndex++) {
        vector<double> layerResults;
        //for every neuron in each layer
        for (unsigned int neuronIndex = 0; neuronIndex < layers[layerIndex].size(); neuronIndex++) {
            auto curN = layers[layerIndex][neuronIndex];
            //calculates neuron output
            double neuronResult = curN->calculate(data);
            double val = ((double) rand() / (RAND_MAX));
            if (val < 1 - chanceDropout) {
                neuronResult = 0;
            }
            curN->prevInputs = data;
            curN->output = neuronResult;
            //adds ReLu activation of neuron calculation to layer results vector
            layerResults.push_back(relu(neuronResult));
        }
        //forward propagates the results to the next layer
        data = layerResults;
    }
    vector<double> newLayerResults;
    //output layer forward prop
    auto outputLayer = layers[layers.size()-1];
    for (unsigned int neuronCount = 0; neuronCount < outputLayer.size(); neuronCount++) {
        auto curN = outputLayer[neuronCount];
        double neuronResult = curN->calculate(data);
        curN->output = neuronResult;
    }
    if (outputLayer.size() == 1) {
        newLayerResults.push_back(outputLayer[0]->output);
    }
    else {
        newLayerResults = softmax();
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

vector<vector<vector<double>>> NeuralNetwork::trainValTestSplit(vector<vector<double>> vec, vector<double> splitRatios) {
    auto trainVec = vectorSplit(vec, 0, ceil(vec.size() * splitRatios[0]));
    auto valVec = vectorSplit(vec, ceil(vec.size() * splitRatios[0]), ceil(vec.size() * (splitRatios[0] + splitRatios[1])));
    auto testVec = vectorSplit(vec, ceil(vec.size() * (splitRatios[0] + splitRatios[1])), ceil(vec.size() - 1));
    return {trainVec, valVec, testVec};
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
        vector<vector<double>> predictedVals;
        for (int i = 0; i < testData.size(); i++) {
            predictedVals.push_back(predictTest(testData[i]));
        }
        return rSquared(predictedVals, testLabel)[1];
    }
}

//Method for predicting a vector representing an unknown data point
vector<double> NeuralNetwork::predict(vector<double> unknownP) {
    vector<vector<double>> reformatUnknown = {unknownP};
    normalize(reformatUnknown);
    auto forwardResult = forwardProp(reformatUnknown[0], 1.0);
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
    auto forwardResult = forwardProp(unknownP, 1.0);
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

//softmax activation function
vector<double> NeuralNetwork::softmax() {
    double denominator = 0;
    vector<double> results;
    auto outLayer = layers[layers.size()-1];
    for (int i = 0; i < outLayer.size(); i++) {
        auto curN = outLayer[i];
        denominator += exp(curN->output);
    }
    for (int i = 0; i < outLayer.size(); i++) {
        auto curN = outLayer[i];
        double result = exp(curN->output) / denominator;
        results.push_back(result);
        layers[layers.size()-1][i]->activatedOutput = result;
    }
    return results;
}

//maybe not needed
vector<double> NeuralNetwork::softmaxDeriv() {
    auto outputLayer = layers[layers.size()-1];
    double denominator = 0;
    for (int i = 0; i < outputLayer.size(); i++) {
        auto curN = outputLayer[i];
        denominator += exp(curN->output);
    }
    vector<double> gradients;
    for (int i = 0; i < outputLayer.size(); i++) {
        auto curN = outputLayer[i];
        gradients.push_back((exp(curN->output) * denominator - pow(exp(curN->output), 2)) / pow(denominator, 2));
    }
    return gradients;
}

//ReLu activation function
double NeuralNetwork::relu(double input) const {
    if (input > 0) {
        return input;
    }
    return 0;
}

//ReLu function derivative, slightly modified
double NeuralNetwork::reluDeriv(double input) const {
    if (input > 0) {
        return 1;
    }
    return 0;
}

//initializes neuron weights to a random value
void NeuralNetwork::initializeWeights(int numWeights, shared_ptr<Neuron> newN, double numOut) {
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

void NeuralNetwork::resetGradients() {
    for (int l = 0; l < layers.size(); l++) {
        for (int n = 0; n < layers[l].size(); n++) {
            layers[l][n]->prevBias = 0;
            for (int g = 0; g < layers[l][n]->prevGradients.size(); g++) {
                layers[l][n]->prevGradients[g] = 0;
            }
        }
    }
}

//gradient descent method for final layer
vector<double> NeuralNetwork:: finalSoftmaxGradient(vector<double> target) {
    vector<double> resultingGradients;
    auto outputLayer = layers[layers.size()-1];
    //auto sDerivs = softmaxDeriv();
    for (int i = 0; i < target.size(); i++) {
        auto curN = outputLayer[i];
        //this didn't work???
        //resultingGradients.push_back(-1 * target[i] * (1 / curN->activatedOutput) * sDerivs[i]);
        //but this thing that looks like MSE does
        resultingGradients.push_back(curN->activatedOutput - target[i]);
    }
    return resultingGradients;
}

double NeuralNetwork::finalLinearGradient(shared_ptr<Neuron> curN, double expected) const {
    return 2 * (curN->output - expected);
}

//gradient descent method for hidden layers
double NeuralNetwork::hiddenGradient(shared_ptr<Neuron> curN, int nIndex, vector<shared_ptr<Neuron>> nextLayer, vector<double> nextDeltas) const {
    double total = 0;
    for (unsigned int i = 0; i < nextLayer.size(); i++) {
        auto newN = nextLayer[i];
        total += newN->weights[nIndex] * nextDeltas[i];
    }
    return reluDeriv(curN->output) * total;
}

//Gets the derivative to be applied to weights
double NeuralNetwork::weightDerivative(double weight, double neuronDelta, double input) const {
    return neuronDelta * input + (2 * this->weightDecay * weight);
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

//displays a progress bar
void NeuralNetwork::progressBar() {
    int barWidth = this->barSize;
    HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
    CONSOLE_CURSOR_INFO cursorInfo;
    GetConsoleCursorInfo(out, &cursorInfo);
    cursorInfo.bVisible = false;
    SetConsoleCursorInfo(out, &cursorInfo);
    while (!*doneTraining) {
        printBar(*curProgress, *progressGoal, barWidth, *timeDuration);
    }
    printBar(1, 1, barWidth, 0);
    cout << endl;
    cursorInfo.bVisible = true;
    SetConsoleCursorInfo(out, &cursorInfo);
}

void NeuralNetwork::printBar(int curVal, int goal, int barWidth, double time) {
    double progress = double(curVal) / goal;
    cout << "\r[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) cout << "=";
        else if (i == pos) cout << ">";
        else cout << " ";
    }
    cout << "] " << std::setw(3) << int(progress * 100.0) << "% | ETA:" << std::setw(3) << int(time / 60) << ":" << std::setfill('0') << std::setw(2) << std::fixed << int(time) % 60 << std::setfill(' ');
    cout.flush();
}

vector<double> NeuralNetwork::rSquared(vector<vector<double>> predicted, vector<vector<double>> target) {
    vector<double> residualVals;
    for (int i = 0; i < predicted.size(); i++) {
        residualVals.push_back(target[i][0] - predicted[i][0]);
    }
    double residualSquared = 0;
    for (int i = 0; i < residualVals.size(); i++) {
        residualSquared += pow(residualVals[i], 2);
    }
    double targetAverage = 0.0;
    for (int i = 0; i < target.size(); i++) {
        targetAverage += target[i][0];
    }
    targetAverage /= target.size();
    double sumSquares = 0;
    for (int i = 0; i < target.size(); i++) {
        sumSquares += pow(target[i][0] - targetAverage, 2);
    }
    double mse = 0.0;
    for (int i = 0; i < residualVals.size(); i++) {
        mse += pow(residualVals[i], 2);
    }
    mse /= residualVals.size();
    return {mse, 1 - (residualSquared / sumSquares)};
}