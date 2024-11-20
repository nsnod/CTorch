#pragma once

#include "array.h"

template <typename T = float>

class Tensor {
 private:
    Array<T>* data_;
    Array<T>* grad_;
    vector<int> shape_;

 public:
    Tensor(vector<int> shape) : shape_(shape), data_(nullptr), grad_(nullptr) {
        if (shape_.size() == 1) {
            shape_.push_back(1);  
        } 

        data_ = new Array<T>(shape_);
        grad_ = new Array<T>(shape_);
    }

    ~Tensor() {
        delete data_;
        delete grad_;
    }
    
    // getters
    Array<T>* getData() { return data_; }
    Array<T>* getGrad() { return data_; }

    // setters
    void setData(Array<T>* inputData_) { data_ = inputData_; }
    void setGrad(Array<T>* inputGrad_) { grad_ = inputGrad_; }


    // Functions 

    //no need for tensor zero. the array is inhertly 0.
    void randomize_tensor(float lower, float upper) {
        // Check if the arrays are initialized properly
        if (data_ != nullptr && grad_ != nullptr) {
            data_->randomize(lower, upper);
            grad_->randomize(lower, upper);
        } else {
            std::cerr << "Error: Tensor arrays not properly initialized!" << std::endl;
        }
    }

    void print_tensor() {
        cout << "Tensor:" << endl;
        cout << "shape" << endl;
        cout << "(";
        for (int i = 0; i < data_->getDim(); i++) { 
            cout << data_->getShape()[i] << ", ";
        }
        cout << ")" << endl << endl;
        cout << "data" << endl;
        data_->print();
        cout << endl;
        cout << "grad" << endl;
        grad_->print();
        cout << endl;
    }

    Tensor<T>& operator+=(T scalar) {
        for (int i = 0; i < data_->getSize(); i++) {
            data_->setData(i, data_->getData()[i] + scalar);
        }

        for (int i = 0; i < grad_->getSize(); i++) {
            grad_->setData(i, grad_->getData()[i] + 1);
        }

        return *this;
    }

    Tensor<T>& operator-=(T scalar) {
        for (int i = 0; i < data_->getSize(); i++) {
            data_->setData(i, data_->getData()[i] - scalar);
        }

        for (int i = 0; i < grad_->getSize(); i++) {
            grad_->setData(i, grad_->getData()[i] - 1);
        }

        return *this;
    }

    Tensor<T>& operator*=(T scalar) {
        
        for (int i = 0; i < data_->getSize(); i++) {
            data_->setData(i, data_->getData()[i] * scalar);
        }

        for (int i = 0; i < grad_->getSize(); i++) {
            grad_->setData(i, grad_->getData()[i] * scalar);
        }

        return *this;
    }


    Tensor<T>& operator/=(T scalar) {
        
        for (int i = 0; i < data_->getSize(); i++) {
            data_->setData(i, data_->getData()[i] / scalar);
        }

        for (int i = 0; i < grad_->getSize(); i++) {
            grad_->setData(i, grad_->getData()[i] / scalar);
        }

        return *this;
    }
    

    // ONLY WORKS FOR 1D AND 2D CURRENTLY
    // in place until we expand for CNN 3ds
    void tensorMulData(Array<T>* data, Array<T>* multData) {
        vector<int> dataShape = data->getShape();
        vector<int> multShape = multData->getShape();
        vector<int> outputShape;

        if (dataShape[dataShape.size() - 1] != multShape[0]) {
            cout << "Your tensors are not able to be multiplied! Check the shape." << endl;
            cout << "Your initial data shape: (";
            for (int i = 0; i < dataShape.size(); i++) {
                cout << dataShape[i] << ", ";
            }
            cout << ")" << endl;

            cout << "Multiplying tensor shape: (";
            for (int i = 0; i < multShape.size(); i++) {
                cout << multShape[i] << ", ";
            }
            cout << ")" << endl;

            cout << endl << "Try again!" << endl;
            return;
        } else {
            outputShape.push_back(dataShape[0]);
            outputShape.push_back(multShape[dataShape.size() - 1]);
        }

        Array<T>* output = new Array<T>(outputShape);

        for (int i = 0; i < dataShape[0]; i++) { 
            for (int j = 0; j < multShape[1]; j++) { 
                T sum = 0;
                for (int k = 0; k < dataShape[1]; k++) { 
                    vector<int> indexA = {i, k}; 
                    vector<int> indexB = {k, j}; 
                    sum += data->at(indexA) * multData->at(indexB);
                }
                vector<int> indexOutput = {i, j}; 
                output->at(indexOutput) = sum; 
            }
        }

        setData(output);
        output = nullptr;
    }

};
