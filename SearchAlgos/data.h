#ifndef DATA_H
#define DATA_H

#include <vector>
#include <string>

// Struct to represent a single row of data
struct DataRow {
    double label;               // Class label (first column)
    std::vector<double> features; // Features (remaining columns)
};

// Function declarations
std::vector<DataRow> loadData(const std::string& filename);
std::vector<DataRow> loadBCData(const std::string& filename);

void normalizeData(std::vector<DataRow>& data);

void zNormalizeData(std::vector<DataRow>& data);

void printData(const std::vector<DataRow>& data);

#endif