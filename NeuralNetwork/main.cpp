#include <iostream>
#include "NeuralNetwork.h"

using std::cout;
using std::endl;
using std::cin;
using std::string;

int main() {
    double learningRate;
    //cout << "Enter learning rate: ";
    //cin >> learningRate;

    //Creates NeuralNetwork "net" with 3 layers, 2 neurons in layer 1, 3 in layer 2, 1 in layer 3
    //sets learning rate to 0.01 (not used yet)
    NeuralNetwork net(4, {3, 10, 5, 3}, 0.01);
    vector<double> expected = {1.0, 0.0, 0.0};
    //Creates a 2d Vector of inputs to test, just one double vector with 3 inputs
    vector<vector<double>> testData = {{0.7, 0.2, 0.0}};
    //still not really used yet, no functioning backprop yet
    auto result = net.forwardProp(testData[0]);
    cout << "Expected: ";
    NeuralNetwork::printVector(expected);
    cout << endl;
    NeuralNetwork::printVector(result);
    cout << endl;
    net.train(testData, expected, 10000);
    result = net.forwardProp(testData[0]);

    result = net.forwardProp(testData[0]);
    NeuralNetwork::printVector(result);
    return 0;
}
