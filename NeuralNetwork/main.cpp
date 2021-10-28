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
    NeuralNetwork net(3, {2, 3, 1}, 0.01);
    //Creates a 2d Vector of inputs to test, just one double vector with 3 inputs
    vector<vector<double>> testData = {{1, 3, 5}};
    //still not really used yet, no functioning backprop yet
    net.train(testData, {0, 0, 1});
    return 0;
}
