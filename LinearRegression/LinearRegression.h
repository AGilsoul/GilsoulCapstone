//
// Created by agils on 10/6/2021.
//
#pragma once

#include <vector>
#include <iostream>
#include <string>

using std::vector;
using std::cout;
using std::endl;
using std::ostream;
using std::string;

class LinearRegression {
public:
    class Point {
    public:
        Point(vector<double> data): data(data) {}
        vector<double> data;
        friend ostream& operator<<(ostream& out, const Point& p) {
            out << "(" << p.data[0];
            for (int i = 1; i < p.data.size(); i++) {
                out << ", " << p.data[i];
            }
            out << ")" << endl;
            return out;
        }
    };

    LinearRegression(int iterations=100000, bool verbose=false): learningRate(0.0001), iterations(iterations), verbose(verbose) {
        this->c = 0;
     }
    LinearRegression(double learningRate, int iterations=100000, bool verbose=false): learningRate(learningRate), iterations(iterations), verbose(verbose) {}

    static void printVector(vector<double>&  v) {
        cout << "{ " << v[0];
        for (int i = 1; i < v.size(); i++) {
                cout << ", " << v[i];
        }
        cout << " }" << endl;
    }


    double predict(Point p);
    double predict(vector<double> p);

    vector<Point> convertData(vector<vector<double>>& v);
    void gradientDescent(vector<Point> x);

    void runTest(vector<Point> testData, vector<Point> trainingData);

    void printResults(double result, string session);
    static vector<Point> vectorSplit(vector<Point> vec, int start, int fin);

private:

    double costFunction(vector<Point> actualOut);
    double derivM(Point actualOut, int index);
    double derivC(Point actualOut);
    double learningRate, c;
    vector<double> coefficients;
    int iterations;
    bool verbose;


};

