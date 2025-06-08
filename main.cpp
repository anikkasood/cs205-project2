
#include <iostream>
#include <fstream>
#include <iomanip>
// measure time
#include <chrono>

//header files
#include "SearchAlgos/search.cpp"
#include "SearchAlgos/data.h"

using namespace std;
using namespace std::chrono;

int main(int argc, char* argv[]) {
    //input which data set to test
    
    string filename;
    int test = 0;
    cout << "Welcome to the Anika Sood Feature Selection Algorithm." << endl;
    cout<< "Type in the name of the file to test : "<<endl; 
    cin>> filename;

    // Load data from file
    vector<DataRow> data = loadData(filename);

    // Get starting timepoint for normalizing data
    auto normalizeStart = high_resolution_clock::now();

    // Normalize data
    zNormalizeData(data);

    // Get ending timepoint for normalizing data
    auto normalizeStop = high_resolution_clock::now();

    //get duration for normalization
    auto normalizationDuration = duration_cast<microseconds>(normalizeStop - normalizeStart);

    // Print normalized data
    // cout << "Normalized Data:" << endl;
    // printData(data);

    cout << "\n\nType the number of the algorithm you want to run.\n" 
         << "(1) Forward Selection\n"  
         << "(2) Backward Elimination\n";

    int featureSelect;
    cin >> featureSelect;

    cout<< "\n\n This dataset has " << data[0].features.size() << " features (not including class attribute), with " 
        << data.size() << " instances." << endl;

    //initialize selection class
    Selection selector(data[0].features.size(), data);

    // Get starting timepoint for computing accuracy
    auto accuracyStart = high_resolution_clock::now();
    ofstream fulloutputFile;

    //output chosen search
    switch(featureSelect) {
        case 1:
        {
            string output = filename.substr(0, filename.length() - 4) + "_forward.txt";
            string fulloutput = filename.substr(0, filename.length() - 4) + "_FULL_forward.txt";
            
            ofstream myfile(output);
            fulloutputFile.open(fulloutput);
           
            selector.forwardSelection(myfile, fulloutputFile);
            
            myfile.close();
            

            
            
            break;
        }
        case 2:
            {
        
            string output = filename.substr(0, filename.length() - 4) + "_backward.txt";
            string fulloutput = filename.substr(0, filename.length() - 4) + "_FULL_backward.txt";
            
            ofstream myfile(output);
            fulloutputFile.open(fulloutput);
           
            selector.backwardElimination(myfile, fulloutputFile);
            
            myfile.close();
            
            break;
        }
        default:
            
            break;
    }

    // Get ending timepoint for computing accuracy
    auto accuracyStop = high_resolution_clock::now();

    //get duration for computing accuracy
    auto accuracyDuration = duration_cast<microseconds>(accuracyStop - accuracyStart);

    //output time for normalization and computing accuracy
    cout << "\nNormalization time       : " << normalizationDuration.count() << " microseconds" << endl;
    cout << "Accuracy Computation time: " << accuracyDuration.count() << " microseconds" << endl;
    
    fulloutputFile << "Normalization time       : " << normalizationDuration.count() << " microseconds" << endl;
    fulloutputFile << "Accuracy Computation time: " << accuracyDuration.count() << " microseconds" << endl;
    fulloutputFile.close();

   
    return 0;
}