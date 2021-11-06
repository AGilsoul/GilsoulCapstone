#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <random>
#include <ctime>
#include "LinearRegression.h"

using std::vector;
using std::string;
using std::ifstream;
using std::ios;
using std::getline;
using std::stod;
using std::stoi;
using std::cout;
using std::endl;
using std::ceil;
using std::cin;
using std::default_random_engine;
using std::time;

vector<vector<double>> readFile2D(string);
vector<vector<double>> readFile3D(string);
void run1(int iterations, double lr);
void run2(int iterations, double lr);
void run3(int iterations, double lr);

int main() {
    int iterations;
    double lr;
    cout << "Enter number of epochs(iterations) to train on:" << endl;
    cin >> iterations;
    cout << "Specify learning rate:" << endl;
    cin >> lr;
    run1(iterations, lr);
    return 0;
}

void run1(int iterations, double lr) {

    vector<vector<double>> tempData = {
            {2.75,2.5,2.5,2.5,2.5,2.5,2.5,2.25,2.25,2.25,2,2,2,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75},
            {5.3,5.3,5.3,5.3,5.4,5.6,5.5,5.5,5.5,5.6,5.7,5.9,6,5.9,5.8,6.1,6.2,6.1,6.1,6.1,5.9,6.2,6.2,6.1},
            {1464,1394,1357,1293,1256,1254,1234,1195,1159,1167,1130,1075,1047,965,943,958,971,949,884,866,876,822,704,719}
    };
    vector<vector<double>> actualData;
    for (int i = 0; i < tempData[0].size(); i++) {
        vector<double> tempV;
        for (int x = 0; x < tempData.size(); x++)  {
            tempV.push_back(tempData[x][i]);
        }
        actualData.push_back(tempV);
    }
    LinearRegression model = LinearRegression(lr, iterations, true);
    auto rng = default_random_engine {};
    rng.seed(time(nullptr));
    shuffle(begin(actualData), end(actualData), rng);
    auto data = model.convertData(actualData);
    auto trainSplit = LinearRegression::vectorSplit(data, 0, ceil(data.size() * 0.6));
    auto testSplit = LinearRegression::vectorSplit(data, ceil(data.size() * 0.6), data.size() - 1);
    model.runTest(testSplit, trainSplit);
}

void run2(int iterations, double lr) {
    auto trainingData = readFile2D("train.csv");
    auto testingData = readFile2D("test.csv");
    LinearRegression model = LinearRegression(lr, iterations, true);
    auto testData = model.convertData(testingData);
    auto practiceData = model.convertData(trainingData);
    model.runTest(testData, practiceData);
    vector<vector<double>> newPoint = {{70, 0}};
    auto newData = model.convertData(newPoint);
    //model.normalize(newData);
    cout << model.predict(newData[0]);
}

void run3(int iterations, double lr) {
    auto tempData = readFile3D("airfoil.csv");
    LinearRegression model = LinearRegression(lr, iterations, true);
    auto data = model.convertData(tempData);
    auto trainSplit = LinearRegression::vectorSplit(data, 0, ceil(data.size() * 0.6));
    auto testSplit = LinearRegression::vectorSplit(data, ceil(data.size() * 0.6), data.size() - 1);
    model.runTest(testSplit, trainSplit);
    vector<vector<double>> newPoint = {{70, 0}};
    auto newData = model.convertData(newPoint);
    //model.normalize(newData);
    cout << model.predict(newData[0]);
}

vector<vector<double>> readFile2D(string fileName) {
    ifstream fin(fileName, ios::in);
    vector<vector<double>> data;
    while (!fin.eof()) {
        vector<double> tempD;
        string sX;
        getline(fin, sX, ',');
        tempD.push_back(stod(sX));
        getline(fin, sX, '\n');
        tempD.push_back(stod(sX));
        data.push_back(tempD);
    }
    fin.close();
    return data;
}

vector<vector<double>> readFile3D(string fileName) {
    ifstream fin(fileName, ios::in);
    vector<vector<double>> data;
    int count = 0;
    while (!fin.eof()) {
        cout << count << endl;
        vector<double> tempD;
        string sX;
        getline(fin, sX, ',');
        tempD.push_back(stod(sX));
        cout << sX << " ";
        getline(fin, sX, ',');
        tempD.push_back(stod(sX));
        cout << sX << " ";
        getline(fin, sX, ',');
        tempD.push_back(stod(sX));
        cout << sX << " ";
        getline(fin, sX, ',');
        tempD.push_back(stod(sX));
        cout << sX << " ";
        getline(fin, sX, ',');
        tempD.push_back(stod(sX));
        cout << sX << " ";
        getline(fin, sX, ',');
        tempD.push_back(stod(sX));
        cout << sX << endl;
        //getline(fin, sX, '\n');
        //tempD.push_back(stod(sX));
        data.push_back(tempD);
        count++;
    }
    fin.close();
    return data;
}
