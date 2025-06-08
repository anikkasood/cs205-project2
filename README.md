# CS 205 Project 2 – Feature Selection with Nearest Neighbor

**Course**: CS 205 Spring 2025  
**Instructor**: Dr. Eamonn Keogh  
**Author**: Anika Sood  
**Repository**: [cs205-project2](https://github.com/anikkasood/cs205-project2)

## Notes

- Forward Selection was in general faster and more accurate across all datasets tested
- Full report is uploaded as PDF to this repository


## Overview

This project implements Forward Selection and Backward Elimination using a Nearest Neighbor classifier on three datasets. The goal is to evaluate how these feature selection methods impact accuracy and computational time.

All code is original, except for some helper functions reused from a previous project in CS 170: [CS170-Project2](https://github.com/anikkasood/CS170-Project2).

## Algorithms

### Forward Selection
Starts with an empty feature set and adds the best-performing feature at each step. At each step, the feature that most improves model accuracy is selected.

### Backward Elimination
Starts with all features and removes the least helpful ones until no further improvement is observed.

## Implementation

The code is divided into three main components:

- `Data`: Reads and normalizes the dataset
- `Classifier`: Implements the Nearest Neighbor classifier
- `Search`: Implements Forward Selection and Backward Elimination

Data is stored using a `DataRow` struct that holds a label and a vector of features.

## Dataset Results

### Small Dataset (`CS205_small_Data__25.txt`)
- 12 features, 500 instances
- Best subset: {3, 5}
- Accuracy: 94.6%
- Forward Selection time: 0.26 minutes
- Backward Elimination time: 0.29 minutes

### Large Dataset (`CS205_large_Data__15.txt`)
- 50 features, 1000 instances
- Forward best subset: {7, 26}, accuracy 96.4%
- Backward best subset: 30-feature subset, accuracy 79.2%
- Forward Selection time: 23.3 minutes
- Backward Elimination time: 30 minutes

### Breast Cancer Wisconsin Dataset
- 30 features, 569 instances
- Preprocessing: removed ID column, mapped 'M' to 1 and 'B' to 0
- Forward best subset: 12 features, accuracy 98.1%
- Backward best subset: 23 features, accuracy 97.5%
- Forward Selection time: 2.5 minutes
- Backward Elimination time: 3.0 minutes

## Why Early Features Matter

In Forward Selection, the first feature selected is the one with the highest individual accuracy. Following features that are selected are in combination with it. Therefore, the intial selection has a great impact on overall accuracy. 

## Hardware

- Processor: 1.4 GHz Quad-Core Intel Core i5  
- Memory: 8 GB RAM  
- Language: C++

## Summary of Computation Times

| Dataset                    | Features / Instances | Forward Selection | Backward Elimination |
|---------------------------|----------------------|-------------------|----------------------|
| Small Dataset             | 12 / 500             | 0.26 min          | 0.29 min             |
| Large Dataset             | 50 / 1000            | 23.3 min          | 30.0 min             |
| Breast Cancer Wisconsin   | 30 / 569             | 2.5 min           | 3.0 min              |

