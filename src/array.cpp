#include "array.h"
#include <iostream>

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
void Array<T>::zeroTensor() {
    fill(data.begin(), data.end(), 0);
}

template <typename T>
int Array<T>::flatIndex(const vector<int>& indices) const {
    int oneDimIndex = 0;
    for (int i = 0; i < dimension; ++i) {
        oneDimIndex += indices[i] * strides[i];
    }
    return oneDimIndex;
}

template <typename T>
T& Array<T>::at(const vector<int>& indices) {
    int oneDimIndex = flatIndex(indices);
    return data[oneDimIndex];
}