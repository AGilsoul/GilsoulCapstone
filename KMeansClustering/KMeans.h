//
// Created by agils on 10/8/2021.
//
#pragma once

#include <vector>
#include <iostream>
#include <stdexcept>
#include <random>
#include <time.h>
#include <chrono>

using std::vector;
using std::string;


class KMeans {
public:
    struct Point {
        Point(vector<double> data): coords(data) {}
        Point(vector<double> data, string label): coords(data), label(label) {}
        vector<double> coords;
        string label = "";
    };

    KMeans(int k);
    void setK(int k);
    int getK();
    static void normalize(vector<Point>& data);
    static vector<Point> vectorSplit(vector<Point> vec, int start, int fin);
    double runTest(const vector<Point>& testData, const vector<Point>& practiceData);
    void train(vector<Point> vec);
    string predictData(vector<double> data);
    vector<Point> static convertData(vector<string> labels, vector<vector<double>> data);



private:
    static int getIndex(vector<string> vec, const string& data);
    static vector<double> sortVector(vector<double> vec);
    static double getDistance(vector<double> data1, vector<double> data2);
    static vector<double> getDistances(vector<double> data, vector<Point> points);
    string predict(vector<double> data) const;
    static double vectorAverage(vector<double> data);

    int k;
    vector<Point> centroids;
    vector<vector<Point>> centroidPoints = {};
    std::mt19937_64 rng;
    std::uniform_real_distribution<double> unif;

};

