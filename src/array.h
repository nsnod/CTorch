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
    void setData(int index, T value) { data[index] = value; }

    // Constructor
    Array(vector<int>& shape) : shape(shape) {
        setDim(shape.size());
        setSize(data.size());
        calcStrides(strides, shape, dimension);
    }

    // Destructor
    ~Array() {}

    // Functions
    void zeroTensor();
    void calcStrides(vector<int>& strides, vector<int> shape, int dimension);
    int flatIndex(const vector<int> indices) const;
    void randomize(float upper, float lower);
    
    // Getter
    int getDim() { return dimension; }
    int getSize() { return size; }
    vector<int> getShape() { return shape; }
    vector<int> getStrides() { return strides; }
    vector<T> getData() { return data; }
    void print();

    T& at(const vector<int>& indices);

    //transpose functionality (inverting shape)
    //create an array of random ints? maybe could be handled in tensor then passed to array?
};