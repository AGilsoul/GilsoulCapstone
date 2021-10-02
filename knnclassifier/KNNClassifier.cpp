// -*- lsst-c++ -*-

/*
 * This file is part of {{ gilsoulcapstone.knnclassifier }}.
 *
 * Developed for the LSST Data Management System.
 * This product includes software developed by the LSST Project
 * (https://www.lsst.org).
 * See the COPYRIGHT file at the top-level directory of this distribution
 * for details of code ownership.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "KNNClassifier.h"
#include <iostream>
#include <vector>
#include <cmath>

using std::string;
using std::vector;
using std::ceil;
using std::pow;


//Public methods
KNNClassifier::KNNClassifier(int k) { this->k = k; }


void KNNClassifier::normalize(vector<Point>& data) {
    for (int p = 0; p < data[0].dblData.size(); p++) {
        vector<double> curData;
        for (auto & i : data) {
            curData.push_back(i.dblData[p]);
        }
        auto sortedData = sortVector(curData);
        for (auto & i : data) {
            i.dblData[p] = (i.dblData[p] - sortedData[0]) / (sortedData[sortedData.size() - 1] - sortedData[0]);
        }
    }
}


vector<Point> KNNClassifier::vectorSplit(vector<Point> vec, int start, int fin) {
    vector<Point> newVec;
    for (int i = start; i <= fin; i++) {
        newVec.push_back(vec[i]);
    }
    return newVec;
}


double KNNClassifier::runTest(const vector<Point>& testData, const vector<Point>& practiceData) {
    double correctGuesses = 0;
    for (auto & tp : testData) {
        string finalGuess = predict(tp.dblData, practiceData);
        if (finalGuess == tp.label) {
            correctGuesses++;
        }
    }
    return correctGuesses / testData.size() * 100;
}


void KNNClassifier::train(vector<Point> vec) {
    this->currentData = std::move(vec);
}


string KNNClassifier::predictData(vector<double> data) {
    return predict(std::move(data), currentData);
}


vector<Point> KNNClassifier::convertData(vector<string> labels, vector<vector<double>> data) {
    vector<Point> points;
    for (int i = 0; i < labels.size(); i++) {
        Point newPoint(labels[i], data[i]);
        points.push_back(newPoint);
    }
    return points;
}


void KNNClassifier::setK(int k) { this->k = k; }


//Private methods
int KNNClassifier::getIndex(vector<string> vec, const string& data) {
    for (int i = 0; i < vec.size(); i++) {
        if (vec[i] == data) {
            return i;
        }
    }
    return -1;
}


vector<double> KNNClassifier::sortVector(vector<double> vec) {
    vector<double> sortedData;
    sortedData.push_back(vec[0]);
    for (int x = 1; x < vec.size(); x++) {
        for (int y = 0; y < sortedData.size(); y++) {
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


vector<double> KNNClassifier::getDistances(vector<double> data, vector<Point> points) {
    vector<double> distances;
    for (auto & point : points) {
        double total = 0;
        for (int l = 0; l < point.dblData.size(); l++) {
            total += pow((point.dblData[l] - data[l]), 2);
        }
        distances.push_back(sqrt(total));
    }
    return distances;
}


string KNNClassifier::predict(vector<double> data, vector<Point> points) const {
    auto distances = getDistances(std::move(data), points);

    vector<Point> sortedPoints;
    vector<double> sortedDistances;
    sortedPoints.push_back(points[0]);
    sortedDistances.push_back(distances[0]);
    for (int i = 1; i < points.size(); i++) {
        for (int x = 0; x < sortedPoints.size(); x++) {
            if (distances[i] < sortedDistances[x]) {
                sortedDistances.insert(sortedDistances.begin() + x, distances[i]);
                sortedPoints.insert(sortedPoints.begin() + x, points[i]);
                break;
            } else if (x == sortedPoints.size() - 1) {
                sortedDistances.push_back(distances[i]);
                sortedPoints.push_back(points[i]);
                break;
            }
        }
    }

    vector<string> guessValues;
    vector<double> guessCounts;
    vector<string> allGuesses;
    for (int i = 0; i < k; i++) {
        string label = sortedPoints[i].label;
        int pos = getIndex(guessValues, label);
        if (pos == -1) {
            guessValues.push_back(label);
            guessCounts.push_back(1);
        } else {
            guessCounts[pos]++;
        }
    }

    double guessMax = 0;
    vector<string> finalGuesses;
    string finalGuess;
    for (int i = 0; i < guessCounts.size(); i++) {
        if (guessCounts[i] > guessMax) {
            guessMax = guessCounts[i];
            finalGuesses = {guessValues[i]};
        } else if (guessCounts[i] == guessMax) {
            finalGuesses.push_back(guessValues[i]);
        }

    }

    if (finalGuesses.size() > 1) {
        int lowest = INT_MAX;
        for (int g = 0; g < finalGuesses.size(); g++) {
            for (int i = 0; i < k; i++) {
                if (finalGuesses[g] == sortedPoints[i].label) {
                    if (i < lowest) {
                        finalGuess = finalGuesses[g];
                        lowest = i;
                    }
                    break;
                }
            }
        }
    } else {
        finalGuess = finalGuesses[0];
    }

    return finalGuess;
}


