#include "KNNClassifier.h"

//Public methods
KNNClassifier::KNNClassifier(int k) { this->k = k; }

void KNNClassifier::normalize(vector<Point>& input) {
    for (int p = 0; p < input[0].data.size(); p++) {
        vector<double> curData;
        for (auto & i : input) {
            curData.push_back(i.data[p]);
        }
        auto sortedData = sortVector(curData);
        for (auto & i : input) {
            i.data[p] = (i.data[p] - sortedData[0]) / (sortedData[sortedData.size() - 1] - sortedData[0]);
        }
    }
}

vector<KNNClassifier::Point> KNNClassifier::vectorSplit(vector<Point> vec, int start, int fin) {
    vector<Point> newVec;
    for (int i = start; i <= fin; i++) {
        newVec.push_back(vec[i]);
    }
    return newVec;
}

double KNNClassifier::runTest(const vector<Point>& testData) {
    double correctGuesses = 0;
    for (auto & tp : testData) {
        string finalGuess = predict(tp.data, currentData);
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

vector<KNNClassifier::Point> KNNClassifier::convertData(vector<string> labels, vector<vector<double>> data) {
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
        for (int l = 0; l < point.data.size(); l++) {
            total += pow((point.data[l] - data[l]), 2);
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