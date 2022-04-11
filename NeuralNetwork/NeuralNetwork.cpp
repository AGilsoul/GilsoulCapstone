//
// Created by Alex Gilsoul on 10/19/2021.
//

#include "../include/NeuralNetwork.h"


// Public Methods

// calculation method for a single neuron
double NeuralNetwork::Neuron::calculate(vector<double> input) {
    // updates the previous input to the neuron
    prevInputs = input;
    double total = 0;
    // calculates the weighted sum
    for (unsigned int w = 0; w < weights.size(); w++) {
        total += weights[w] * input[w];
    }
    // adds the neuron bias
    total += bias;
    return total;
}

// neural network constructor, takes number of hidden layers + output layer, and neuron counts for each
NeuralNetwork::NeuralNetwork(vector<int> neurons, double learningRate, double momentum, bool verbose, int barSize) {
    this->learningRate = learningRate;
    this->momentum = momentum;
    this->verbose = verbose;
    this->barSize = barSize;
    int numLayers = neurons.size();
    uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed>>32)};
    rng.seed(ss);

    // initializes neurons and weights for every layer except input layer
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

// min-max data normalization method
void NeuralNetwork::normalize(vector<vector<double>>& input, vector<double> minMaxRange) {
    // if conversion rates don't already exist
    if (!conversions) {
        // if the range of all data values is known and the same for all data traits
        if (minMaxRange.size() > 0) {
            // for every data trait
            for (unsigned int traitIndex = 0; traitIndex < input[0].size(); traitIndex++) {
                // vector containing all values in the dataset for the current trait
                vector<double> D;
                for (auto &x: input) {
                    D.push_back(x[traitIndex]);
                }
                // gets the min and max values for the data range
                double min = minMaxRange[0];
                double max = minMaxRange[1];
                // for every data point x in the dataset
                for (auto &x: input) {
                    // applies min-max normalization
                    x[traitIndex] = (x[traitIndex] - min) / (max - min);
                    // if the value is undefined (max and min are the same), set to 0
                    if (std::isnan(x[traitIndex])) {
                        x[traitIndex] = 0;
                    }
                }
                // saves the data range for later conversions
                vector<double> tempFactors = {min, max};
                conversionRates.push_back(tempFactors);
            }
        }
        // if the range of all data values is unknown or different for data traits
        else {
            // for every data trait
            for (unsigned int traitIndex = 0; traitIndex < input[0].size(); traitIndex++) {
                // vector containing all values in the dataset for the current trait
                vector<double> D;
                for (auto &x: input) {
                    D.push_back(x[traitIndex]);
                }
                // gets the min and max values for the data range
                auto limits = vectorMinMax(D);
                double min = limits[0];
                double max = limits[1];
                // for every data point x in the dataset
                for (auto &x: input) {
                    // applies min-max normalization
                    x[traitIndex] = (x[traitIndex] - min) / (max - min);
                    // if the value is undefined (max and min are the same), set to 0
                    if (std::isnan(x[traitIndex])) {
                        x[traitIndex] = 0;
                    }
                }
                // saves the data range for later conversions
                vector<double> tempFactors = {min, max};
                conversionRates.push_back(tempFactors);
            }
        }
        // confirms that conversion rates are saved
        conversions = true;
    }
    // if conversion rates do exist
    else {
        // for every data trait
        for (unsigned int traitIndex = 0; traitIndex < input[0].size(); traitIndex++) {
            // for every data point x in the dataset
            for (auto &x: input) {
                // applies min-max normalization
                x[traitIndex] = (x[traitIndex] - conversionRates[traitIndex][0]) / (conversionRates[traitIndex][1] - conversionRates[traitIndex][0]);
                // if the value is undefined (max and min are the same), set to 0
                if (std::isnan(x[traitIndex])) {
                    x[traitIndex] = 0;
                }
            }
        }
    }
}

// loads saved weight data
void NeuralNetwork::loadData(string fileName) {
    ifstream fin(fileName, ios::in);
    string numLayers, nPL, lr, m;
    // gets learning rate
    std::getline(fin, lr, ',');
    learningRate = stod(lr);
    // gets momentum
    std::getline(fin, m, '\n');
    momentum = stod(m);
    // gets number of layers
    std::getline(fin, numLayers, '\n');
    vector<int> neuronsPerLayer;
    // gets neuron counts for each layer
    for (unsigned int i = 0; i < stoi(numLayers) - 1; i++) {
        std::getline(fin, nPL, ',');
        neuronsPerLayer.push_back(stoi(nPL));
    }
    std::getline(fin, nPL, '\n');
    neuronsPerLayer.push_back(stoi(nPL));
    // for every neuron, gets weight count and weights
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

// writes testing results to a csv file
bool NeuralNetwork::writeResults(string fileName, vector<double> primKey, vector<double> results, vector<double> runtime) {
    // try writing to file
    ofstream saveFile(fileName);
    try {
        saveFile << "PrimaryKey,Accuracy,Runtime" << endl;
        for (int i = 0; i < primKey.size(); i++) {
            saveFile << primKey[i] << "," << results[i] << "," << runtime[i];
            if (i != primKey.size() - 1) {
                saveFile << endl;
            }
        }
        return true;
    }
    catch(const std::exception& e) {
        cout << e.what() << endl;
        return false;
    }
}

// for performance testing, modify as needed
void NeuralNetwork::perTest(vector<vector<double>> input, vector<vector<double>> allResults, vector<vector<double>> testIn, vector<vector<double>> testResults, int iterations, string fOut) {
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
    std::thread thread_obj(&NeuralNetwork::_perTest, *this, input, allResults, testIn, testResults, iterations, fOut);
    if (verbose) {
        std::thread thread_progress(&NeuralNetwork::progressBar, *this);
        thread_progress.join();
    }
    thread_obj.join();
}

void NeuralNetwork::_perTest(vector<vector<double>> input, vector<vector<double>> allResults, vector<vector<double>> testIn, vector<vector<double>> testResults, int iterations, string fOut) {
    double lr = learningRate;
    double m = momentum;
    vector<double> primKeys;
    vector<double> trainAccuracies;
    vector<double> testAccuracies;
    vector<double> runtimes;
    double runtime = 0;
    StopWatch watch;
    *progressGoal = iterations * input.size();
    // for every iteration
    for (unsigned int z = 0; z < iterations; z++) {
        // for every training data point
        for (unsigned int x = 0; x < input.size(); x++) {
            watch.reset();
            *curProgress += 1;
            // gets the actual result of the current data point
            auto desiredResult = allResults[x];
            // gets predicted result from forward propagation
            vector<double> finalResult = forwardProp(input[x], this->dropOutRate);
            // sets up the nextDelta variables for the hidden layers
            vector<double> nextDeltas;
            // output layer back propagation
            if (layers[layers.size()-1].size() == 1) {
                for (unsigned int neuronCount = 0; neuronCount < layers[layers.size() - 1].size();neuronCount++) {
                    // current neuron
                    auto curN = layers[layers.size()- 1][neuronCount];
                    // gets the derivative of the neuron with respect to the expected output
                    curN->delta = finalLinearGradient(curN, desiredResult[neuronCount]);
                    // adds the delta to the nextDeltas vector
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

            // hidden layer backprop for every hidden layer
            for (int layerCount = layers.size() - 2; layerCount >= 0; layerCount--) {
                // tempDeltas vector, will be the nextDeltas vector for the previous layer
                vector<double> tempDeltas;
                // for every neuron in the hidden layer
                for (unsigned int neuronCount = 0; neuronCount < layers[layerCount].size(); neuronCount++) {
                    // current neuron
                    auto curN = layers[layerCount][neuronCount];
                    // gets the derivative of the neuron with respect to the next layer neurons
                    curN->delta = hiddenGradient(curN, neuronCount, layers[layerCount + 1], nextDeltas);
                    tempDeltas.push_back(curN->delta);
                }
                nextDeltas = tempDeltas;
            }

            // updating weights in every layer
            for (int layerCount = layers.size() - 1; layerCount >= 0; layerCount--) {
                // for every neuron in the layer
                for (unsigned int neuronCount = 0; neuronCount < layers[layerCount].size(); neuronCount++) {
                    auto curN = layers[layerCount][neuronCount];
                    // updates every weight and previous gradient for the current neuron
                    for (int w = 0; w < curN->weights.size(); w++) {
                        // gets the derivative of weight adjust with the delta of the current neuron and the inputs
                        double result = weightDerivative(curN->delta, curN->prevInputs[w]) * lr;
                        curN->weights[w] -= result + curN->prevGradients[w] * m;
                        curN->prevGradients[w] = result + curN->prevGradients[w] * m;
                    }
                    // updates bias and previous bias for the current neuron
                    double bResult = curN->delta * lr;
                    curN->bias -= bResult + curN->prevBias * m;
                    curN->prevBias = bResult + curN->prevBias * m;
                }
            }
            runtime += watch.elapsed_time();
            if ((z*input.size() + x) % 1000 == 0) {
                cout << z*input.size() + x << endl;
                primKeys.push_back(z*input.size() + x);
                testAccuracies.push_back(test(testIn, testResults));
                trainAccuracies.push_back(test(input, allResults));
                runtimes.push_back(runtime);
            }
        }
    }
    string testOut = "test" + fOut;
    string trainOut = "train" + fOut;
    cout << writeResults(trainOut, primKeys, trainAccuracies, runtimes) << endl;
    cout << writeResults(testOut, primKeys, testAccuracies, runtimes) << endl;
    *doneTraining = true;
    *loadedData = true;
}


// back propagation method, repeats for every iteration
void NeuralNetwork::trainWithValidation(vector<vector<double>> trainInput, vector<vector<double>> trainResults,vector<vector<double>> valInput, vector<vector<double>> valResults, int minIterations, int maxIterations) {
    *curProgress = 0;
    *progressGoal = 1;
    *doneTraining = false;
    *validationIters = 0;
    // initialize input neuron weights
    if (!*loadedData) {
        for (unsigned int i = 0; i < layers[0].size(); i++) {
            initializeWeights(trainInput[0].size(), layers[0][i], layers[1].size());
        }
    }
    else {
        resetGradients();
    }
    if (this->verbose) cout << "SGD Progress:" << endl;
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
    // for every iteration
    for (unsigned int z = 0; z < maxIterations; z++) {
        *validationIters += 1;
        // for every training data point
        for (unsigned int x = 0; x < trainInput.size(); x++) {
            *curProgress += 1;
            // gets the actual result of the current data point
            auto desiredResult = trainResults[x];
            // gets predicted result from forward propagation
            vector<double> finalResult = forwardProp(trainInput[x], this->dropOutRate);
            // sets up the nextDelta variables for the hidden layers
            vector<double> nextDeltas;
            // output layer back propagation
            if (layers[layers.size()-1].size() == 1) {
                for (unsigned int neuronCount = 0; neuronCount < layers[layers.size() - 1].size();neuronCount++) {
                    // current neuron
                    auto curN = layers[layers.size()- 1][neuronCount];
                    // gets the derivative of the neuron with respect to the expected output
                    curN->delta = finalLinearGradient(curN, desiredResult[neuronCount]);

                    // adds the delta to the nextDeltas vector
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


            // hidden layer backprop for every hidden layer
            for (int layerCount = layers.size() - 2; layerCount >= 0; layerCount--) {
                // tempDeltas vector, will be the nextDeltas vector for the previous layer
                vector<double> tempDeltas;
                // for every neuron in the hidden layer
                for (unsigned int neuronCount = 0; neuronCount < layers[layerCount].size(); neuronCount++) {
                    // current neuron
                    auto curN = layers[layerCount][neuronCount];
                    // gets the derivative of the neuron with respect to the next layer neurons
                    curN->delta = hiddenGradient(curN, neuronCount, layers[layerCount + 1], nextDeltas);
                    tempDeltas.push_back(curN->delta);
                }
                nextDeltas = tempDeltas;
            }

            // updating weights in every layer
            for (int layerCount = layers.size() - 1; layerCount >= 0; layerCount--) {
                // for every neuron in the layer
                for (unsigned int neuronCount = 0; neuronCount < layers[layerCount].size(); neuronCount++) {
                    auto curN = layers[layerCount][neuronCount];
                    // updates every weight and previous gradient for the current neuron
                    for (int w = 0; w < curN->weights.size(); w++) {
                        // gets the derivative of weight adjust with the delta of the current neuron and the inputs
                        double result = weightDerivative(curN->delta, curN->prevInputs[w]) * lr;
                        curN->weights[w] -= result + curN->prevGradients[w] * m;
                        curN->prevGradients[w] = result + curN->prevGradients[w] * m;
                    }
                    // updates bias and previous bias for the current neuron
                    double bResult = curN->delta * lr;
                    curN->bias -= bResult + curN->prevBias * m;
                    curN->prevBias = bResult + curN->prevBias * m;
                }
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

// back propagation method, repeats for every iteration
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
    // for every iteration
    for (unsigned int z = 0; z < iterations; z++) {
        // for every training data point
        for (unsigned int x = 0; x < input.size(); x++) {
            *curProgress += 1;
            // gets the actual result of the current data point
            auto desiredResult = allResults[x];
            // gets predicted result from forward propagation
            vector<double> finalResult = forwardProp(input[x], this->dropOutRate);
            // sets up the nextDelta variables for the hidden layers
            vector<double> nextDeltas;
            // output layer back propagation
            if (layers[layers.size()-1].size() == 1) {
                for (unsigned int neuronCount = 0; neuronCount < layers[layers.size() - 1].size();neuronCount++) {
                    // current neuron
                    auto curN = layers[layers.size()- 1][neuronCount];
                    // gets the derivative of the neuron with respect to the expected output
                    curN->delta = finalLinearGradient(curN, desiredResult[neuronCount]);
                    // adds the delta to the nextDeltas vector
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

            // hidden layer backprop for every hidden layer
            for (int layerCount = layers.size() - 2; layerCount >= 0; layerCount--) {
                // tempDeltas vector, will be the nextDeltas vector for the previous layer
                vector<double> tempDeltas;
                // for every neuron in the hidden layer
                for (unsigned int neuronCount = 0; neuronCount < layers[layerCount].size(); neuronCount++) {
                    // current neuron
                    auto curN = layers[layerCount][neuronCount];
                    // gets the derivative of the neuron with respect to the next layer neurons
                    curN->delta = hiddenGradient(curN, neuronCount, layers[layerCount + 1], nextDeltas);
                    tempDeltas.push_back(curN->delta);
                }
                nextDeltas = tempDeltas;
            }

            // updating weights in every layer
            for (int layerCount = layers.size() - 1; layerCount >= 0; layerCount--) {
                // for every neuron in the layer
                for (unsigned int neuronCount = 0; neuronCount < layers[layerCount].size(); neuronCount++) {
                    auto curN = layers[layerCount][neuronCount];
                    // updates every weight and previous gradient for the current neuron
                    for (int w = 0; w < curN->weights.size(); w++) {
                        // gets the derivative of weight adjust with the delta of the current neuron and the inputs
                        double result = weightDerivative(curN->delta, curN->prevInputs[w]) * lr;
                        curN->weights[w] -= result + curN->prevGradients[w] * m;
                        curN->prevGradients[w] = result + curN->prevGradients[w] * m;
                    }
                    // updates bias and previous bias for the current neuron
                    double bResult = curN->delta * lr;
                    curN->bias -= bResult + curN->prevBias * m;
                    curN->prevBias = bResult + curN->prevBias * m;
                }
            }
        }
    }
    *doneTraining = true;
    *loadedData = true;
}

// back propagation method, repeats for every iteration
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
    // for every iteration
    for (unsigned int z = 0; z < iterations; z++) {
        vector<vector<vector<double>>> batches;
        vector<vector<vector<double>>> batchResults;

        int numBatches = floor(input.size() / batchSize);

        vector<int> indexes;
        indexes.reserve(input.size());
        for (int i = 0; i < input.size(); ++i)
            indexes.push_back(i);

        std::random_shuffle(indexes.begin(), indexes.end());

        // for every batch
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


        // for every batch
        int timeDurations = 0;
        for (unsigned int batchCount = 0; batchCount < batches.size(); batchCount++) {
            // reset neuron deltas
            for (int la = 0; la < layers.size(); la++) {
                for (int na = 0; na < layers[la].size(); na++) {
                    layers[la][na]->delta = 0;
                }
            }
            // for every training data point
            for (unsigned int x = 0; x < batches[batchCount].size(); x++) {
                *curProgress += 1;
                // gets the actual result of the current data point
                auto desiredResult = batchResults[batchCount][x];
                // gets predicted result from forward propagation
                vector<double> finalResult = forwardProp(batches[batchCount][x], this->dropOutRate);
                // sets up the nextDelta variables for the hidden layers
                vector<double> nextDeltas;
                // output layer back propagation
                if (layers[layers.size()-1].size() == 1) {
                    for (unsigned int neuronCount = 0; neuronCount < layers[layers.size() - 1].size();neuronCount++) {
                        // current neuron
                        auto curN = layers[layers.size()- 1][neuronCount];
                        // gets the derivative of the neuron with respect to the expected output
                        curN->delta += finalLinearGradient(curN, desiredResult[neuronCount]);

                        // adds the delta to the nextDeltas vector
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

                // hidden layer backprop for every hidden layer
                for (int layerCount = layers.size() - 2; layerCount >= 0; layerCount--) {
                    // tempDeltas vector, will be the nextDeltas vector for the previous layer
                    vector<double> tempDeltas;
                    // for every neuron in the hidden layer
                    for (unsigned int neuronCount = 0; neuronCount < layers[layerCount].size(); neuronCount++) {
                        // current neuron
                        auto curN = layers[layerCount][neuronCount];
                        // gets the derivative of the neuron with respect to the next layer neurons
                        curN->delta += hiddenGradient(curN, neuronCount, layers[layerCount + 1], nextDeltas);
                        tempDeltas.push_back(curN->delta);
                    }
                    nextDeltas = tempDeltas;
                }
            }
            // updating weights in every layer
            for (int layerCount = layers.size() - 1; layerCount >= 0; layerCount--) {
                // for every neuron in the layer
                for (unsigned int neuronCount = 0; neuronCount < layers[layerCount].size(); neuronCount++) {
                    auto curN = layers[layerCount][neuronCount];
                    curN->delta /= batches[batchCount].size();
                    // updates every weight and previous gradient for the current neuron
                    for (int w = 0; w < curN->weights.size(); w++) {
                        // gets the derivative of weight adjust with the delta of the current neuron and the inputs
                        double result = weightDerivative(curN->delta, curN->prevInputs[w]) * lr;
                        curN->weights[w] -= (result + curN->prevGradients[w] * m);
                        curN->prevGradients[w] = (result + curN->prevGradients[w] * m);
                    }
                    // updates bias and previous bias for the current neuron
                    double bResult = curN->delta * lr;
                    curN->bias -= bResult + curN->prevBias * m;
                    curN->prevBias = bResult + curN->prevBias * m;
                }
            }
        }
    }
    *doneTraining = true;
    *loadedData = true;
}

// back propagation method, repeats for every iteration
void NeuralNetwork::trainMiniBatchValidation(vector<vector<double>> trainInput, vector<vector<double>> trainResults, vector<vector<double>> valInput, vector<vector<double>> valResults, int minIterations, int maxIterations, int batchSize) {
    *curProgress = 0;
    *progressGoal = 1;
    *doneTraining = false;
    *validationIters = 0;
    // initialize input neuron weights
    if (!*loadedData) {
        for (unsigned int i = 0; i < layers[0].size(); i++) {
            initializeWeights(trainInput[0].size(), layers[0][i], layers[1].size());
        }
    }
    else {
        resetGradients();
    }
    if (this->verbose) cout << "Mini-Batch Progress:" << endl;
    std::thread thread_obj(&NeuralNetwork::_trainMiniBatchValidation, *this, trainInput, trainResults, valInput, valResults, minIterations, maxIterations, batchSize);
    if (verbose) {
        std::thread thread_progress(&NeuralNetwork::progressBar, *this);
        thread_progress.join();
    }
    thread_obj.join();
    cout << "Trained for " << *validationIters << " iterations" << endl;
}

void NeuralNetwork::_trainMiniBatchValidation(vector<vector<double>> trainInput, vector<vector<double>> trainResults, vector<vector<double>> valInput, vector<vector<double>> valResults, int minIterations, int maxIterations, int batchSize) {
    double lr = learningRate;
    double m = momentum;
    double prevAccuracy = 0.0;
    int iterationsDecreased = 0;
    *progressGoal = maxIterations * trainInput.size();
    // for every iteration
    for (unsigned int z = 0; z < maxIterations; z++) {
        *validationIters += 1;
        vector<vector<vector<double>>> batches;
        vector<vector<vector<double>>> batchResults;

        int numBatches = floor(trainInput.size() / batchSize);

        vector<int> indexes;
        indexes.reserve(trainInput.size());
        for (int i = 0; i < trainInput.size(); ++i)
            indexes.push_back(i);

        std::random_shuffle(indexes.begin(), indexes.end());

        // for every batch
        for (unsigned int i = 0; i < numBatches; i++) {
            vector<vector<double>> curBatch;
            vector<vector<double>> curResults;
            for (unsigned int x = i * batchSize; x < (i * batchSize) + batchSize; x++) {
                curBatch.push_back(trainInput[indexes[x]]);
                curResults.push_back(trainResults[indexes[x]]);
            }
            batches.push_back(curBatch);
            batchResults.push_back(curResults);
        }


        // for every batch
        int timeDurations = 0;
        for (unsigned int batchCount = 0; batchCount < batches.size(); batchCount++) {
            // reset neuron deltas
            for (int la = 0; la < layers.size(); la++) {
                for (int na = 0; na < layers[la].size(); na++) {
                    layers[la][na]->delta = 0;
                }
            }
            // for every training data point
            for (unsigned int x = 0; x < batches[batchCount].size(); x++) {
                *curProgress += 1;
                // gets the actual result of the current data point
                auto desiredResult = batchResults[batchCount][x];
                // gets predicted result from forward propagation
                vector<double> finalResult = forwardProp(batches[batchCount][x], this->dropOutRate);
                // sets up the nextDelta variables for the hidden layers
                vector<double> nextDeltas;
                // output layer back propagation
                if (layers[layers.size()-1].size() == 1) {
                    for (unsigned int neuronCount = 0; neuronCount < layers[layers.size() - 1].size();neuronCount++) {
                        // current neuron
                        auto curN = layers[layers.size()- 1][neuronCount];
                        // gets the derivative of the neuron with respect to the expected output
                        curN->delta += finalLinearGradient(curN, desiredResult[neuronCount]);

                        // adds the delta to the nextDeltas vector
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

                // hidden layer backprop for every hidden layer
                for (int layerCount = layers.size() - 2; layerCount >= 0; layerCount--) {
                    // tempDeltas vector, will be the nextDeltas vector for the previous layer
                    vector<double> tempDeltas;
                    // for every neuron in the hidden layer
                    for (unsigned int neuronCount = 0; neuronCount < layers[layerCount].size(); neuronCount++) {
                        // current neuron
                        auto curN = layers[layerCount][neuronCount];
                        // gets the derivative of the neuron with respect to the next layer neurons
                        curN->delta += hiddenGradient(curN, neuronCount, layers[layerCount + 1], nextDeltas);
                        tempDeltas.push_back(curN->delta);
                    }
                    nextDeltas = tempDeltas;
                }
            }
            // updating weights in every layer
            for (int layerCount = layers.size() - 1; layerCount >= 0; layerCount--) {
                // for every neuron in the layer
                for (unsigned int neuronCount = 0; neuronCount < layers[layerCount].size(); neuronCount++) {
                    auto curN = layers[layerCount][neuronCount];
                    curN->delta /= batches[batchCount].size();
                    // updates every weight and previous gradient for the current neuron
                    for (int w = 0; w < curN->weights.size(); w++) {
                        // gets the derivative of weight adjust with the delta of the current neuron and the inputs
                        double result = weightDerivative(curN->delta, curN->prevInputs[w]) * lr;
                        curN->weights[w] -= (result + curN->prevGradients[w] * m);
                        curN->prevGradients[w] = (result + curN->prevGradients[w] * m);
                    }
                    // updates bias and previous bias for the current neuron
                    double bResult = curN->delta * lr;
                    curN->bias -= bResult + curN->prevBias * m;
                    curN->prevBias = bResult + curN->prevBias * m;
                }
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

// forward propagation method
vector<double> NeuralNetwork::forwardProp(vector<double> input, double chanceDropout) {
    auto data = input;
    // for every hidden layer and input layer
    for (int layerIndex = 0; layerIndex < layers.size() - 1; layerIndex++) {
        vector<double> layerResults;
        // for every neuron in each layer
        for (unsigned int neuronIndex = 0; neuronIndex < layers[layerIndex].size(); neuronIndex++) {
            auto curNeuron = layers[layerIndex][neuronIndex];
            // calculates neuron output
            double neuronResult = curNeuron->calculate(data);
            double val = ((double) rand() / (RAND_MAX));
            if (val < 1 - chanceDropout) {
                neuronResult = 0;
            }
            curNeuron->prevInputs = data;
            curNeuron->output = neuronResult;
            // adds ReLu activation of neuron calculation to layer results vector
            layerResults.push_back(relu(neuronResult));
        }
        // forward propagates the results to the next layer
        data = layerResults;
    }
    vector<double> newLayerResults;
    // output layer forward prop
    auto outputLayer = layers[layers.size()-1];
    for (unsigned int neuronCount = 0; neuronCount < outputLayer.size(); neuronCount++) {
        auto curNeuron = outputLayer[neuronCount];
        double neuronResult = curNeuron->calculate(data);
        curNeuron->output = neuronResult;
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

// static method for printing vectors of doubles
void NeuralNetwork::printVector(vector<double> input) {
    cout << "{ " << input[0];
    for (unsigned int i = 1; i < input.size(); i++) {
        cout << ", " << input[i];
    }
    cout << " }";
}

// splits up a 2D vector of doubles based on specified start/end indices
vector<vector<double>> NeuralNetwork::vectorSplit(vector<vector<double>> vec, int start, int fin) {
    vector<vector<double>> newVec;
    for (unsigned int i = start; i <= fin; i++) {
        newVec.push_back(vec[i]);
    }
    return newVec;
}

// splits up a 2D vector of doubles based on specified split ratios
vector<vector<vector<double>>> NeuralNetwork::trainValTestSplit(vector<vector<double>> vec, vector<double> splitRatios) {
    auto trainVec = vectorSplit(vec, 0, ceil(vec.size() * splitRatios[0]));
    auto valVec = vectorSplit(vec, ceil(vec.size() * splitRatios[0]), ceil(vec.size() * (splitRatios[0] + splitRatios[1])));
    auto testVec = vectorSplit(vec, ceil(vec.size() * (splitRatios[0] + splitRatios[1])), ceil(vec.size() - 1));
    return {trainVec, valVec, testVec};
}

// testing method, compares predicted results after training with actual results
double NeuralNetwork::test(vector<vector<double>>& testData, vector<vector<double>>& testLabel) {
    if (layers[layers.size() - 1].size() > 1) {
        double accuracy = 0;
        // for every test data point
        for (unsigned int i = 0; i < testData.size(); i++) {
            // gets forward propagation result with current test data point
            auto newResults = predictTest(testData[i]);
            bool correct = true;
            // compares predicted answer with actual answer
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

// method for predicting a vector representing an unknown data point
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

// resets the weights of all neurons in the network
void NeuralNetwork::resetWeights(int dataCount) {
    for (int lc = 0; lc < layers.size(); lc++) {
        // normalized xavier weight initialization
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

// private Methods
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

// softmax activation function
vector<double> NeuralNetwork::softmax() {
    // sets the denominator of the softmax function to 0
    double denominator = 0;
    // vector to contain the activated outputs
    vector<double> results;
    // creates a copy of the output layer
    auto outLayer = layers[layers.size()-1];
    // for every neuron in the output layer
    for (int n = 0; n < outLayer.size(); n++) {
        // creates a reference to the current neuron
        auto curNeuron = outLayer[n];
        // adds e raised to the current neuron output to the denominator
        denominator += exp(curNeuron->output);
    }
    // for every neuron in the output layer
    for (int n = 0; n < outLayer.size(); n++) {
        // creates a reference to the current neuron
        auto curN = outLayer[n];
        // calculates the activated output for the current neuron
        double result = exp(curN->output) / denominator;
        // adds the activated output to the result vector
        results.push_back(result);
        // sets the activated output of the current neuron
        layers[layers.size()-1][n]->activatedOutput = result;
    }
    return results;
}

// ReLu activation function
double NeuralNetwork::relu(double output) const {
    // if the output is greater than 0, return the output
    if (output > 0) {
        return output;
    }
    // else return 0
    return 0;
}

// ReLu function derivative, slightly modified
double NeuralNetwork::reluDeriv(double output) const {
    // if the neuron output is greater than 0, return 1
    if (output > 0) {
        return 1;
    }
    // else return 0
    return 0;
}

// initializes neuron weights to a random value
void NeuralNetwork::initializeWeights(int numWeights, shared_ptr<Neuron> newN, double numOut) {
    // normalized xavier weight initialization
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

// resets all gradients
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

// gets gradient of final softmax gradient
vector<double> NeuralNetwork:: finalSoftmaxGradient(vector<double> target) {
    // vector to store the gradients
    vector<double> resultingGradients;
    // creates a copy of the output layer
    auto outputLayer = layers[layers.size()-1];
    // for every category in the target/output neuron
    for (int t = 0; t < target.size(); t++) {
        // creates a reference to the current neuron
        auto curN = outputLayer[t];
        // this didn't work???
        // resultingGradients.push_back(-1 * target[t] * (1 / curN->activatedOutput) * sDerivs[t]);
        // but this thing that looks like MSE does
        // add resulting derivative to the gradients vector
        resultingGradients.push_back(curN->activatedOutput - target[t]);
    }
    return resultingGradients;
}

// gets gradient of final regression layer
double NeuralNetwork::finalLinearGradient(shared_ptr<Neuron> curN, double expected) const {
    return 2 * (curN->output - expected);
}

// gets gradient in hidden layers
double NeuralNetwork::hiddenGradient(shared_ptr<Neuron> curN, int nIndex, vector<shared_ptr<Neuron>> nextLayer, vector<double> nextDeltas) const {
    double total = 0;
    for (unsigned int i = 0; i < nextLayer.size(); i++) {
        auto newN = nextLayer[i];
        total += newN->weights[nIndex] * nextDeltas[i];
    }
    return reluDeriv(curN->output) * total;
}

// gets the derivative to be applied to weights
double NeuralNetwork::weightDerivative(double neuronDelta, double input) const {
    return neuronDelta * input;
}

// finds the min and max value of a vector
vector<double> NeuralNetwork::vectorMinMax(vector<double> vec) {
    double min;
    double max;
    for (unsigned int i = 0; i < vec.size(); i++) {
        if (vec[i] < min || !min) min = vec[i];
        else if (vec[i] > max || !max) max = vec[i];
    }
    return {min, max};
}

// saves weight data to csv file called "nn_save_config.csv"
bool NeuralNetwork::saveData(string fileName) {
    ofstream saveFile(fileName);
    if (saveFile) {
        // writes learning rate and momentum
        saveFile << learningRate << "," << momentum << "\n";
        // writes number of layers to the top
        saveFile << layers.size() << "\n";
        // writes neuron counts for each layer to the top
        for (unsigned int lCount = 0; lCount < layers.size() - 1; lCount++) {
            saveFile << layers[lCount].size() << ",";
        }
        saveFile << layers[layers.size() - 1].size() << "\n";
        for (unsigned int lCount = 0; lCount < layers.size(); lCount++) {
            // writes number of weights and weights underneath
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

// displays a progress bar
void NeuralNetwork::progressBar() {
    int barWidth = this->barSize;
    HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
    CONSOLE_CURSOR_INFO cursorInfo;
    GetConsoleCursorInfo(out, &cursorInfo);
    cursorInfo.bVisible = false;
    SetConsoleCursorInfo(out, &cursorInfo);
    StopWatch progressWatch;
    progressWatch.reset();
    int startTime = progressWatch.elapsed_time();
    while (!*doneTraining) {
        printBar(*curProgress, *progressGoal, barWidth, progressWatch, startTime);
    }
    printBar(1, 1, barWidth, progressWatch, startTime);
    cout << endl;
    cursorInfo.bVisible = true;
    SetConsoleCursorInfo(out, &cursorInfo);
}

// helper for progressBar method
void NeuralNetwork::printBar(int curVal, int goal, int barWidth, StopWatch watch, int startTime) {
    double progress = double(curVal) / goal;
    cout << "\r[";
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) cout << "=";
        else if (i == pos) cout << ">";
        else cout << " ";
    }
    cout << "] " << std::setw(3) << int(progress * 100.0) << "% " << loading[int(watch.elapsed_time()) % 4];
    cout.flush();
}

// calculates r-squared values for regression
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

// calculates average cross entropy loss for classification
double NeuralNetwork::crossEntropy(vector<vector<double>> predicted, vector<vector<double>> target) {
    // sets average to 0
    double average = 0;
    // for every output neuron/target
    for (int x = 0; x < predicted.size(); x++) {
        // for every category
        for (int i = 0; i < predicted[0].size(); i++) {
            average -= log2(predicted[x][i]) * target[x][i];
        }
    }
    return average / predicted.size();
}