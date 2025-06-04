#include <iostream>
#include <cstdlib>
#include <ctime>

#include <string>
#include <vector>

#include <cmath>
#include <limits>

#include "data.h"  // Assuming you have DataRow defined in this header
using namespace std;

class Classifier {
private:
    vector<DataRow> training;  // Store training data

    DataRow test;              // Store test data

public:
    // Constructor with optional parameters to initialize training data and test data
    Classifier(const vector<DataRow>& trainingData = {}, const DataRow& testData = {}) {
        training = trainingData;  // Initialize training data
        test = testData;          // Initialize test data
    }

    // Training function that takes training data as argument
    void train(const vector<DataRow>& trainingData) {
        cout << "Training the classifier..." << endl;
        training = trainingData;  // Store the provided training data
    }

     double calc_dist(const DataRow& a, const DataRow& b) {
        double sum = 0.0;
        for (size_t i = 0; i < a.features.size(); ++i) {
            double diff = a.features[i] - b.features[i];
            sum += diff * diff;
        }
        return sqrt(sum);  // Euclidean distance
    }

    // Method to predict the label of the test row
    int predict(const DataRow& test_row, const vector<DataRow>& training_data) {
        double min_dist = numeric_limits<double>::max();
        int predicted_label = -1;

        // Find the closest training row
        for (const auto& training_row : training_data) {
            double dist = calc_dist(training_row, test_row);
            if (dist < min_dist) {
                min_dist = dist;
                predicted_label = training_row.label;
            }
        }
        return predicted_label;
    }
};