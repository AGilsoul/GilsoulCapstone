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

#pragma once

#include "Point.h"
#include <iostream>
#include <utility>
#include <vector>

using std::vector;
using std::string;
using std::ostream;


class KNNClassifier {
public:
    /**
     * Default constructor for KNNClassifier
     *
     * @param k the specified 'k' value for the algorithm
     */
    explicit KNNClassifier(int k);

    /**
     *Normalizes values vectors containing doubles for each Point in a vector
     *
     * @param[out] data
     */
    static void normalize(vector<Point>& data);

    /**
     * Returns a subvector of a vector containing Point instances
     *
     * @param vec the vector to to be split
     * @param start the starting index of the subvector
     * @param fin the final index of the subvector
     * @return subvector of the original given vector
     */
    static vector<Point> vectorSplit(vector<Point> vec, int start, int fin);

    /**
     *Uses predict method to test all points in testData and returns percent correct
     *
     * @param testData vector of Point instances to be tested on
     * @param practiceData vector of Point instances to use for predictions
     * @return percentage accuracy of points correctly identified
     */
    double runTest(const vector<Point>& testData, const vector<Point>& practiceData);

    /**
     * Gives the classifier object data to use for future predictions
     *
     * @param vec vector of Point instances to use for future predictions
     */
    void train(vector<Point> vec);

    /**
     * Predicts label of a collecton of data based on labels of nearest Point objects using the Point collection
     * that the classifier was trained on
     *
     * @param data the data collection whose label will be predicted
     * @return a string of the predicted label
     */
    string predictData(vector<double> data);

    /**
     * Given a collection of labels and a collection of data, converts each corresponding label and data to a Point
     *
     * @param labels labels for each data point
     * @param data a collection of vectors of data for each data point
     * @return a vector containing all of the newly created Points
     */
    vector<Point> static convertData(vector<string> labels, vector<vector<double>> data);

    /**
     * Changes the current 'k' value of the classifier
     * @param k integer representing how many Points will be considered when predicting data labels
     */
    void setK(int k);

private:
    /**
     * Returns the index of a string in a collection of strings
     *
     * @param vec the collection to be searched
     * @param data the string to search for
     * @return the index of the string within the collection, returns -1 if not found
     */
    static int getIndex(vector<string> vec, const string& data);

    /**
     * Sorts a vector from minimum to maximum using a selection sort algorithm
     *
     * @param vec the vector to be sorted
     * @return a sorted vector
     */
    static vector<double> sortVector(vector<double> vec);

    /**
     * Returns a collection the Euclidean distances of each point in a collection to a collection of data
     *
     * @param data data to find distance from
     * @param points the Points whose distance from the data will be found
     * @return a vector of doubles containing the distance of each point to the data collection
     */
    static vector<double> getDistances(vector<double> data, vector<Point> points);

    /**
     * Given a data point (vector of doubles) and a vector of Points, predicts the label of the data point by
     * sorting the Points and finding the most common label of the 'k' nearest Points
     *
     * @param data data point whose label will be predicted
     * @param points collection of Points to predict the data's label with
     * @return a string of the predicted label
     */
    string predict(vector<double> data, vector<Point> points) const;

    int k;
    vector<Point> currentData;
};


