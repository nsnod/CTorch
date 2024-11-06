#pragma once

#include <vector>

using namespace std;

template <typename T>
class Array {
private:
    vector<int> shape;
    vector<T> data;
    vector<int> strides;
    int dimension;
    int size;

public:
    // Setter
    void setDim(int dim) { dimension = dim; }
    void setSize(int s) { size = s; }

    // Constructor
    Array(vector<T>& data, vector<int>& shape) : data(data), shape(shape) {
        setDim(shape.size());
        setSize(data.size());
        calcStrides(strides, shape, dimension);
    }

    // Destructor
    ~Array() {}

    // Functions
    void zeroTensor();
    void calcStrides(vector<int>& strides, vector<int> shape, int dimension);
    int flatIndex(const vector<int>& indices) const;
    
    // Getter
    T& at(const vector<int>& indices);

    //transpose functionality (inverting shape)
    //create an array of random ints? maybe could be handled in tensor then passed to array?
};