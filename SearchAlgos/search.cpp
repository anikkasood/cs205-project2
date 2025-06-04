#include <iostream>
#include <set>
#include <cstdlib>

#include <ctime>
#include <iomanip>

#include "data.h"
using namespace std;

#include "classifi.cpp"

class Selection {

private:
    int num_features;

    double best_accuracy;

    set<int> best_subset;

    vector<DataRow> data;

public:
    //constructor
    Selection(int num_features, vector<DataRow> data) : num_features(num_features), best_accuracy(0.0), data(data) {}

    //evaluation function
    double eval(const set<int>& features, const vector<DataRow>& data_a00) {
        vector<DataRow> data_0;
        Classifier classifier;
        // Prepare the data by extracting the features specified in 'features'
        for (const auto& row : data_a00) {
            DataRow new_row = row;
            vector<double> features_test;

            for (const int feature_index : features) {
                if (feature_index >= 0 && feature_index - 1 < new_row.features.size()) {
                    features_test.push_back(new_row.features[feature_index - 1]);  // Adjusted for 1-based indexing
                }
            }

            new_row.features = features_test;
            data_0.push_back(new_row);
        }

        int correct_predictions = 0;

        // Perform leave-one-out cross-validation
        for (size_t i = 0; i < data_0.size(); ++i) {
            DataRow test_data = data_0[i];

            // Prepare the training data by leaving out the current test data
            vector<DataRow> training_data;
            for (size_t j = 0; j < data_0.size(); ++j) {
                if (j != i) {
                    training_data.push_back(data_0[j]);
                }
            }

            // Predict the label for the test data
            int predicted_label = classifier.predict(test_data, training_data);

            // Compare the predicted label with the true label
            if (predicted_label == test_data.label) {
                ++correct_predictions;
            }
        }

        // Calculate and return the accuracy
        double accuracy = static_cast<double>(correct_predictions) / data_0.size();
        return accuracy * 100;  // Return the accuracy as a percentage
    }

    // Forward selection algorithm
    void forwardSelection() {
        cout << fixed << setprecision(1); // for 1 decimal place

        set<int> cur_features;
        cout << "Forward Selection:" << endl;
        cout << "Running nearest neighbor with all 4 features, using “leaving-one-out” evaluation, I get an accuracy of "<< 
            eval(cur_features, this->data) << "%" << endl;
        
            cout << "Beginning search.\n" << endl;

        for (int i = 1; i <= num_features; ++i) {
            double best_accuracy_for_round = 0.0;
            int best_feature = -1;


            for (int feature = 1; feature <= num_features; ++feature) {
                if (cur_features.find(feature) == cur_features.end()) {
                    set<int> test_features = cur_features;
                    test_features.insert(feature);

                    double accuracy = eval(test_features, this->data);
                    cout << "Using feature(s) {";
                    for (auto it = test_features.begin(); it != test_features.end(); ++it) {
                        cout << *it;
                        if (next(it) != test_features.end()) cout << ",";
                    }
                    cout << "} accuracy is " << accuracy << "%" << endl;

                    if (accuracy > best_accuracy_for_round) {
                        best_accuracy_for_round = accuracy;
                        best_feature = feature;
                    }
                }
            }

            if (best_feature != -1) {
                cur_features.insert(best_feature);
                cout << "Feature set {";
                for (auto it = cur_features.begin(); it != cur_features.end(); ++it) {
                    cout << *it;
                    if (next(it) != cur_features.end()) cout << ",";
                }
                cout << "} was best, accuracy is "  << best_accuracy_for_round << "%\n" << endl;

                
                if (best_accuracy_for_round > best_accuracy) {
                    best_accuracy = best_accuracy_for_round;
                    best_subset = cur_features;
                }
            } 
        }

        cout << "Finished search!! The best feature subset is {";
        for (auto it = best_subset.begin(); it != best_subset.end(); ++it) {
            cout << *it;
            if (next(it) != best_subset.end()) cout << ",";
        }
        cout << "}, which has an accuracy of "  << best_accuracy << "%" << endl;
    }

    //backward selection algorithm
    void backwardElimination() {
        set<int> cur_features;
        cout << fixed << setprecision(1); // for 1 decimal place

        // make the set of all of the features
        for (int i = 1; i <= num_features; ++i) {
            cur_features.insert(i);
        }

        cout << "Backward Elimination:" << endl;
        cout << "Using all features and 'random' evaluation, I get an accuracy of "
             << eval(cur_features, this->data) << "%" << endl;
        cout << "Beginning search.\n" << endl;

        // set all features as the best accuracy for now
        best_accuracy = eval(cur_features, this->data);
        best_subset = cur_features; // to track the subset for the best feature 

        while (cur_features.size() > 1) {
            double best_accuracy_for_round = 0.0;
            int feature_to_remove = -1;

            for (int feature : cur_features) {
                set<int> test_features = cur_features; // store all of the features within a set
                test_features.erase(feature); // remove the curr one

                double accuracy = eval(test_features, this->data); // get the accuracy of everything BUT the feature just eliminated
                cout << "Using feature(s) {";
                for (auto it = test_features.begin(); it != test_features.end(); ++it) {
                    cout << *it;
                    if (next(it) != test_features.end()) cout << ",";
                }
                cout << "} accuracy is " << accuracy << "%" << endl;

                 // if its the best we have so far for this level make this elimination + update
                if (accuracy > best_accuracy_for_round) {
                    best_accuracy_for_round = accuracy;
                    feature_to_remove = feature;
                }
            }
            
            // display the best for that round
            if (feature_to_remove != -1) {
                cur_features.erase(feature_to_remove);
                cout << "Feature set {";
                for (auto it = cur_features.begin(); it != cur_features.end(); ++it) {
                    cout << *it;
                    if (next(it) != cur_features.end()) cout << ",";
                }
                cout << "} was best, accuracy is " << best_accuracy_for_round << "%\n" << endl;

                // if its better than the overall best, update the overall best
                if (best_accuracy_for_round > best_accuracy) { 
                    best_accuracy = best_accuracy_for_round;
                    best_subset = cur_features;
                }
            }
        }

        // disp overall best
        cout << "Finished search!! The best feature subset is {";
        for (auto it = best_subset.begin(); it != best_subset.end(); ++it) {
            cout << *it;
            if (next(it) != best_subset.end()) cout << ",";
        }
        cout << "}, which has an accuracy of " << best_accuracy << "%" << endl;
    }
};