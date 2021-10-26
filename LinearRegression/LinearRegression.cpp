//
// Created by agils on 10/6/2021.
//

#include "LinearRegression.h"
#include <cmath>
#include <string>
#include <iostream>

using std::cout;
using std::endl;
using std::string;

vector<LinearRegression::Point> LinearRegression::convertData(vector<vector<double>>& v) {
    vector<Point> pData;
    for (int i = 0; i < v.size(); i++) {
        Point newP(v[i]);
        pData.push_back(newP);
    }

    return pData;
}

double LinearRegression::predict(Point p) {
    double total = 0;
    for (int i = 0; i < coefficients.size(); i++) {
        total += coefficients[i] * p.data[i];
    }
    double result = total + c;
    return result;
}


double LinearRegression::predict(vector<double> p) {
    double total = 0;
    for (int i = 0; i < coefficients.size(); i++) {
        total += coefficients[i] * p[i];
    }
    double result = total + c;
    return result;
}


//partial derivative of loss function, with respect to m
//Dm = (-2/n)(summation n terms, i = 0,
void LinearRegression::gradientDescent(vector<Point> input) {
    for (int i = 0; i < input[0].data.size() - 1; i++) {
        coefficients.push_back(0);
    }
    int iterator = 0;
    while (iterator < iterations) {
        for (int m = 0; m < coefficients.size(); m++) {
            coefficients[m] -= learningRate * derivM(input, coefficients[m], m);
        }
        c -= learningRate * derivC(input);
        iterator++;
        if (std::isnan(c) || std::isnan(sqrt(costFunction(input)))) {
            throw std::invalid_argument("Learning rate of " + std::to_string(learningRate)+ " is too high");
        }
        if (verbose) {
            cout << "Iteration #" << iterator << endl;
            cout << "Coefficient(s): ";
            printVector(coefficients);
            cout << "Intercept: " << c << endl << endl;
        }
    }
}


void LinearRegression::runTest(vector<Point> testData, vector<Point> trainingData) {
    gradientDescent(trainingData);
    double resultTrain = costFunction(trainingData);
    double resultTest = costFunction(testData);
    if (verbose) {
        printResults(resultTrain, "TRAINING");
        printResults(resultTest, "**TEST**");
    }

}

void LinearRegression::printResults(double result, string session) {
    cout << endl << "**********************" << session << "**********************" << endl;
    cout << "Mean Squared Error: " << result << endl;
    cout << "Mean Error: " << sqrt(result) << endl;
    cout << "Coefficient(s): {" << coefficients[0];
    for (int i = 1; i < coefficients.size(); i++) {
        cout << ", " << coefficients[i];
    }
    cout << "} c: " << c << endl;
}


vector<LinearRegression::Point> LinearRegression::vectorSplit(vector<Point> vec, int start, int fin) {
    vector<Point> newVec;
    for (int i = start; i <= fin; i++) {
        newVec.push_back(vec[i]);
    }
    return newVec;
}


// mean error = (1/n)(summation n terms, i = 0 (yi - (mxi + c))^2
double LinearRegression::costFunction(vector<Point> actualOut) {
    double total = 0;
    for (int i = 0; i < actualOut.size(); i++) {
        int result = pow(predict(actualOut[i]) - actualOut[i].data[actualOut[i].data.size() - 1], 2);
        total += result;
    }
    return (total / actualOut.size());
}
//Cost function partial derivative with respect to m
double LinearRegression::derivM(vector<Point> input, double coefficient, int index) {
    double total = 0;
    for (int i = 0; i < input.size(); i++) {
        double result = 2 * (input[i].data[input[i].data.size() - 1] - (predict(input[i]))) * -input[i].data[index];
        total += result;
    }
    return total / input.size();
}
//Cost function partial derivative with respect to c
double LinearRegression::derivC(vector<Point> input) {
    double total = 0;
    for (int i = 0; i < input.size(); i++) {
        double prediction = predict(input[i]);
        total += -2 * (input[i].data[input[i].data.size() - 1] - prediction);
    }
    return total / input.size();
}
