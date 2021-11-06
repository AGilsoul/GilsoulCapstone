//
// Created by agils on 10/8/2021.
//

#include "KMeans.h"

KMeans::KMeans(int k) {
    this->k = k;
    // initialize the random number generator with time-dependent seed
    uint64_t timeSeed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::seed_seq ss{uint32_t(timeSeed & 0xffffffff), uint32_t(timeSeed>>32)};
    this->rng.seed(ss);
    // initialize a uniform distribution between 0 and 1
    this->unif = std::uniform_real_distribution<double>(0, 1);
}

void KMeans::setK(int k) {
    this->k = k;
}

int KMeans::getK() {
    return k;
}


void KMeans::normalize(vector<Point>& data) {
    for (int p = 0; p < data[0].coords.size(); p++) {
        vector<double> curData;
        for (auto & i : data) {
            curData.push_back(i.coords[p]);
        }
        auto sortedData = sortVector(curData);
        double median = sortedData[ceil(sortedData.size() / 2)];
        double p75 = sortedData[ceil(sortedData.size() / 2) + ceil(sortedData.size() / 4)];
        double p25 = sortedData[ceil(sortedData.size() / 4)];
        //for (auto & i : data) {
            //i.coords[p] = (i.coords[p] - sortedData[0]) / (sortedData[sortedData.size() - 1] - sortedData[0]);
        //}
        for (auto & i : data) {
            i.coords[p] = (i.coords[p] - median) / (p75 - p25);
        }
    }

}


vector<KMeans::Point> KMeans::vectorSplit(vector<Point> vec, int start, int fin) {
    vector<Point> newVec;
    for (int i = start; i <= fin; i++) {
        newVec.push_back(vec[i]);
    }
    return newVec;
}


double KMeans::runTest(const vector<Point>& testData, const vector<Point>& practiceData) {
    double correctGuesses = 0;
    for (auto & tp : testData) {
        string finalGuess = predict(tp.coords);
        if (finalGuess == tp.label) {
            correctGuesses++;
        }
    }
    return correctGuesses / testData.size() * 100;
}


void KMeans::train(vector<Point> input) {
    centroids.clear();
    vector<string> labels;
    vector<vector<Point>> labelPoints;
    // ready to generate random numbers
    for (int i = 0; i < k; i++) {
        vector<double> tempCoords;
        for (int dataCount = 0; dataCount < input[0].coords.size(); dataCount++) {
            tempCoords.push_back(unif(rng));
        }
        Point newP(tempCoords);
        centroids.push_back(newP);
    }
    vector<double> centroidDiffs;
    bool converged = false;
    vector<Point> prevCentroids;

    //do while loop, updates centroid locations
    do {
        centroidPoints.clear();
        centroidDiffs.clear();
        prevCentroids = centroids;
        for (int i = 0; i < k; i++) {
            vector<Point> tempVec;
            centroidPoints.push_back(tempVec);
        }
        //assign points to nearest centroids
        for (int inputCount = 0; inputCount < input.size(); inputCount++) {
            vector<double> cDist = getDistances(input[inputCount].coords, centroids);
            int minIndex;
            double minVal = INT_MAX;
            for (int dist = 0; dist < cDist.size(); dist++) {
                if (cDist[dist] < minVal) {
                    minVal = cDist[dist];
                    minIndex = dist;
                }
            }
            centroidPoints[minIndex].push_back(input[inputCount]);
        }

        //Move centroids based on average assigned point distance
        for (int centroidCount = 0; centroidCount < centroids.size(); centroidCount++) {
            //saves previous centroid data
            vector<double> newData;
            //for every data index
            if (centroidPoints[centroidCount].size() != 0) {
                for (int column = 0; column < centroidPoints[centroidCount][0].coords.size(); column++) {
                    //for every centroid's assigned data points
                    double colTotal = 0;
                    for (int inputCount = 0; inputCount < centroidPoints[centroidCount].size(); inputCount++) {
                        colTotal += centroidPoints[centroidCount][inputCount].coords[column];
                    }

                    newData.push_back(colTotal / centroidPoints[centroidCount].size());
                }
                //assigns new data to centroid
                centroids[centroidCount].coords = newData;
                centroidDiffs.push_back(getDistance(centroids[centroidCount].coords, prevCentroids[centroidCount].coords));
            }

        }

        bool allZero = true;
        for (int centroidCount = 0; centroidCount < centroidDiffs.size(); centroidCount++) {
            if (centroidDiffs[centroidCount] != 0) {
                allZero = false;
            }
        }
        if (allZero) {
            converged = true;
        }
    } while (!converged);

    //determine centroid labels based on data points
    for (int centroidCount = 0; centroidCount < centroids.size(); centroidCount++) {
        if (centroidPoints[centroidCount].size() != 0) {
            vector<string> labels;
            vector<double> labelCounts;
            for (int dataP = 0; dataP < centroidPoints[centroidCount].size(); dataP++) {
                string curLabel = centroidPoints[centroidCount][dataP].label;
                int index = getIndex(labels, curLabel);
                if (index == -1) {
                    labels.push_back(curLabel);
                    labelCounts.push_back(1);
                }
                else {
                    labelCounts[index]++;
                }
            }
            int maxIndex;
            int maxCount = 0;
            for (int labelNum = 0; labelNum < labels.size(); labelNum++) {
                if (labelCounts[labelNum] > maxCount) {
                    maxCount = labelCounts[labelNum];
                    maxIndex = labelNum;
                }
            }
            centroids[centroidCount].label = labels[maxIndex];
        }
    }
    for (int centroidCount = 0; centroidCount < centroids.size(); centroidCount++) {
        if (centroidPoints[centroidCount].size() == 0) {
            vector<double> distances = getDistances(centroids[centroidCount].coords, centroids);
            int minDist = INT_MAX;
            int index;
            for (int i = 0; i < distances.size(); i++) {
                if (distances[i] < minDist && distances[i] != 0 && centroids[i].label != "") {
                    minDist = distances[i];
                    index = i;
                }
            }
            centroids[centroidCount].label = centroids[index].label;
        }
    }
    std::cout << "Centroid Labels: ";
    for (int i = 0; i < centroids.size(); i++) {
        std:: cout << centroids[i].label << " ";
    }
    std:: cout << std::endl;
}


string KMeans::predictData(vector<double> data) {
    return predict(std::move(data));
}


vector<KMeans::Point> KMeans::convertData(vector<string> labels, vector<vector<double>> data) {
    vector<Point> points;
    for (int i = 0; i < labels.size(); i++) {
        KMeans::Point newPoint(data[i], labels[i]);
        points.push_back(newPoint);
    }
    return points;
}


int KMeans::getIndex(vector<string> vec, const string& data) {
    for (int i = 0; i < vec.size(); i++) {
        if (vec[i] == data) {
            return i;
        }
    }
    return -1;
}


vector<double> KMeans::sortVector(vector<double> vec) {
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


double KMeans::getDistance(vector<double> data1, vector<double> data2) {
    double total = 0;
    for (int l = 0; l < data1.size(); l++) {
        total += pow((data2[l] - data1[l]), 2);
    }
    return sqrt(total);
}


vector<double> KMeans::getDistances(vector<double> data, vector<Point> points) {
    vector<double> distances;
    for (auto & point : points) {
        double total = 0;
        for (int l = 0; l < point.coords.size(); l++) {
            total += pow((point.coords[l] - data[l]), 2);
        }
        distances.push_back(sqrt(total));
    }
    return distances;
}


string KMeans::predict(vector<double> data) const {
    vector<double> distances = getDistances(data, centroids);
    int lowestDistance = INT_MAX;
    int lowestIndex = INT_MAX;
    for (int i = 0; i < distances.size(); i++) {
        if (distances[i] < lowestDistance) {
            lowestDistance = distances[i];
            lowestIndex = i;
        }
    }

    return centroids[lowestIndex].label;

}

double KMeans::vectorAverage(vector<double> data) {
    double total = 0.0;
    for (int i = 0; i < data.size(); i++) {
        total += data[i];
    }
    return total / data.size();
}

