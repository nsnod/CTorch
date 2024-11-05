#pragma once

#include <vector>

using namespace std;

template <typename T>

class Array{
  private:
    vector<int> shape;
    vector<T> data;
    vector<int> strides;
    int dimension;
    int size;

    void calcStrides() {
        strides.resize(dimension);
        strides[dimension - 1] = 1;
        // - 2 instead of - 1 because the last dimension will always be one move iterable. no skipping needed
        for (int i = dimension - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
    }

    Array(vector<int> shape) : shape(shape) {
        dimension = shape.size();
        size = 1;

        for (int dim : shape) {
            size *= dim;
        }

        data.resize(size, 0); 

        calcStrides();
    }

    Array(vector<T> data, vector<int> shape) : data(data), shape(shape) {
        dimension = shape.size();

        size = data.size();

        calcStrides();
    }

    ~Array() {

    }

    int flatIndex(const vector<int>& indices) const {
        //try and catch if the indices is the same as the size of stride
        int oneDimIndex = 0;
        for (int i = 0; i < dimension; ++i) {
            index += indices[i] * strides[i];
        }
        return oneDimIndex;
    }

    //getter
    T& at(const vector<int>& indices) {
        int oneDimIndex = flatIndex(indices);
        return data[oneDimIndex];
    }


    //transpose functionality (inverting shape)
    //create an array of random ints? maybe could be handled in tensor then passed to array?
};