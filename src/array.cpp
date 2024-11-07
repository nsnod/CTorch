#include "array.h"
#include <iostream>
#include <random>

using namespace std;

template <typename T>
void Array<T>::calcStrides(vector<int>& strides, vector<int> shape, int dimension) {
    strides.resize(dimension);
    strides[dimension - 1] = 1;
    for (int i = dimension - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
}

template <typename T>
void Array<T>::randomize(float upper, float lower) {
    random_device rd; 
    mt19937 gen(rd()); 
    uniform_real_distribution<float> dist(lower, upper); 
    for (int i = 0; i < getSize(); i++) {
        setData(i, dist(gen));
    }
}

// work on
template <typename T>
void Array<T>::print() {
    vector<T> data = getData();
    vector<int> indicies(getShape().size(), 0);
    int right = indicies.size() - 1;
    int left = indicies.size() - 1;
    for (int i = 0; i < indicies.size(); i++) {
        while (right >= left && left >= 0) {
            for (int i = 0; i < right; i++) {
                
            }
            left--;
            right = indicies.size() - 1;
        }
    }
        
    // 3 2 2

    // 0 0 0
    // 0 0 1
    // 0 1 0
    // 0 1 1
    // 1 0 0
    // 2 0 0
    // 2 0 1
    // 2 1 0
    // 2 1 1

    
}

template <typename T>
int Array<T>::flatIndex(const vector<int> indices) const {
    int oneDimIndex = 0;
    for (int i = 0; i < dimension; ++i) {
        oneDimIndex += indices[i] * strides[i];
    }
    return oneDimIndex;
}

template <typename T>
T& Array<T>::at(const vector<int>& indices) {
    int oneDimIndex = flatIndex(indices);
    return getData()[oneDimIndex];
}