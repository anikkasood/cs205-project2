#include <iostream>
#include <fstream>
#include <set>
#include <cstdlib>
#include <ostream>
#include <ctime>
#include <string>       
#include <sstream>     

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

    string format1Decimal(double value) {
        ostringstream oss;
        oss << fixed << setprecision(1) << value;
        return oss.str();
    }



    // Forward selection algorithm
    void forwardSelection(ofstream & myfile, ofstream & fulloutput) {

        vector<string> file_output;

        cout << fixed << setprecision(1); // for 1 decimal place
        



        set<int> cur_features;
        cout << "Forward Selection:" << endl;
        file_output.push_back("Forward Selection:");

        double accuracy = eval(cur_features, this->data);
        cout << "Running nearest neighbor with all 4 features, using “leaving-one-out” evaluation, I get an accuracy of "<< 
            accuracy << "%" << endl;
        file_output.push_back("Running nearest neighbor with all 4 features, using “leaving-one-out” evaluation, I get an accuracy of " + format1Decimal(accuracy) + "%");
            cout << "Beginning search.\n" << endl;

        file_output.push_back("Beginning search.");

        for (int i = 1; i <= num_features; ++i) {
            double best_accuracy_for_round = 0.0;
            int best_feature = -1;


            for (int feature = 1; feature <= num_features; ++feature) {
                if (cur_features.find(feature) == cur_features.end()) {
                    set<int> test_features = cur_features;
                    test_features.insert(feature);
                    
                    string subset_str = "{";

                    double accuracy = eval(test_features, this->data);
                    cout << "Using feature(s) {";
                    for (auto it = test_features.begin(); it != test_features.end(); ++it) {
                        cout << *it;
                        subset_str += to_string(*it);

                        if (next(it) != test_features.end()) {
                            cout << ",";
                            subset_str += ","; // update str for output file
                        }
                    }
                    subset_str += "}";

                    cout << "} accuracy is " << accuracy << "%" << endl;
                    myfile << subset_str << " " << accuracy << endl;
                    file_output.push_back("Using feature(s)" + subset_str + " " + format1Decimal(accuracy));

                    if (accuracy > best_accuracy_for_round) {
                        best_accuracy_for_round = accuracy;
                        best_feature = feature;
                    }
                }
            }

            if (best_feature != -1) {
                cur_features.insert(best_feature);
                
              string toAddToFile = "Feature set {";
                cout << "Feature set {";
                for (auto it = cur_features.begin(); it != cur_features.end(); ++it) {
                    cout << *it;
                    toAddToFile += to_string(*it); // update str for output file

                    if (next(it) != cur_features.end()) {
                        cout << ",";
                        toAddToFile +=  ",";
                    }
                }
               
                cout << "} was best, accuracy is "  << best_accuracy_for_round << "%\n" << endl;
                toAddToFile += "} was best, accuracy is " + format1Decimal(best_accuracy_for_round) + "%";
                file_output.push_back(toAddToFile);

                if (best_accuracy_for_round > best_accuracy) {
                    best_accuracy = best_accuracy_for_round;
                    best_subset = cur_features;
                }
            } 
        }

        string toAddToFile = "Finished search!! The best feature subset is {";
        cout << "Finished search!! The best feature subset is {";
        for (auto it = best_subset.begin(); it != best_subset.end(); ++it) {
            cout << *it;
            toAddToFile += to_string(*it); // update str for output file
            
            if (next(it) != best_subset.end()) {
                cout << ",";
                toAddToFile += ",";
            }
        }
        cout << "}, which has an accuracy of "  << best_accuracy << "%" << endl;
        toAddToFile += "}, which has an accuracy of " + format1Decimal(best_accuracy) + "%";
        file_output.push_back(toAddToFile);

        int n = file_output.size();
        
        if (n <= 100) {
            for (const string& line : file_output)
                fulloutput << line << '\n';
        } else {
            for (int i = 0; i < 50; ++i)
                fulloutput << file_output[i] << '\n';

            fulloutput << "...\n... [omitted " << (n - 100) << " lines] ...\n...\n";

            for (int i = n - 50; i < n; ++i)
                fulloutput << file_output[i] << '\n';
        }

    }
//backward selection algorithm
void backwardElimination(ofstream & myfile, ofstream & fulloutput) {
    vector<string> file_output;
    set<int> cur_features;
    cout << fixed << setprecision(1); // for 1 decimal place

    string initial_subset = "{";
    for (int i = 1; i <= num_features; ++i) {
        cur_features.insert(i);
        initial_subset += to_string(i);
        if (i < num_features) {
            initial_subset += ",";
        }
    }
    initial_subset += "}";

    myfile << initial_subset << " ";
    double init = eval(cur_features, this->data);

    cout << "Backward Elimination:" << endl;
    file_output.push_back("Backward Elimination:");

    cout << "Using all features and 'random' evaluation, I get an accuracy of " << init << "%" << endl;
    file_output.push_back("Using all features and 'random' evaluation, I get an accuracy of " + format1Decimal(init) + "%");

    myfile << init << endl;
    cout << "Beginning search.\n" << endl;
    file_output.push_back("Beginning search.");

    best_accuracy = eval(cur_features, this->data);
    best_subset = cur_features;

    while (cur_features.size() > 1) {
        double best_accuracy_for_round = 0.0;
        int feature_to_remove = -1;

        for (int feature : cur_features) {
            set<int> test_features = cur_features;
            test_features.erase(feature);

            string subset_str = "{";
            double accuracy = eval(test_features, this->data);
            cout << "Using feature(s) {";
            for (auto it = test_features.begin(); it != test_features.end(); ++it) {
                cout << *it;
                subset_str += to_string(*it);
                if (next(it) != test_features.end()) {
                    cout << ",";
                    subset_str += ",";
                }
            }
            subset_str += "}";
            cout << "} accuracy is " << accuracy << "%" << endl;
            myfile << subset_str << " " << accuracy << endl;
            file_output.push_back("Using feature(s) " + subset_str + " " + format1Decimal(accuracy) + "%");

            if (accuracy > best_accuracy_for_round) {
                best_accuracy_for_round = accuracy;
                feature_to_remove = feature;
            }
        }

        if (feature_to_remove != -1) {
            cur_features.erase(feature_to_remove);
            string toAddToFile = "Feature set {";
            cout << "Feature set {";
            for (auto it = cur_features.begin(); it != cur_features.end(); ++it) {
                cout << *it;
                toAddToFile += to_string(*it);
                if (next(it) != cur_features.end()) {
                    cout << ",";
                    toAddToFile += ",";
                }
            }
            cout << "} was best, accuracy is " << best_accuracy_for_round << "%\n" << endl;
            toAddToFile += "} was best, accuracy is " + format1Decimal(best_accuracy_for_round) + "%";
            file_output.push_back(toAddToFile);

            if (best_accuracy_for_round > best_accuracy) {
                best_accuracy = best_accuracy_for_round;
                best_subset = cur_features;
            }
        }
    }

    string toAddToFile = "Finished search!! The best feature subset is {";
    cout << "Finished search!! The best feature subset is {";
    for (auto it = best_subset.begin(); it != best_subset.end(); ++it) {
        cout << *it;
        toAddToFile += to_string(*it);
        if (next(it) != best_subset.end()) {
            cout << ",";
            toAddToFile += ",";
        }
    }
    cout << "}, which has an accuracy of " << best_accuracy << "%" << endl;
    toAddToFile += "}, which has an accuracy of " + format1Decimal(best_accuracy) + "%";
    file_output.push_back(toAddToFile);

    int n = file_output.size();
    if (n <= 100) {
        for (const string& line : file_output)
            fulloutput << line << '\n';
    } else {
        for (int i = 0; i < 50; ++i)
            fulloutput << file_output[i] << '\n';
        fulloutput << "...\n... [omitted " << (n - 100) << " lines] ...\n...\n";
        for (int i = n - 50; i < n; ++i)
            fulloutput << file_output[i] << '\n';
    }
}

};