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
        // Initialize the member variables (not local variables)
        data_ = new Array<T>(shape_);
        grad_ = new Array<T>(shape_);
    }

    ~Tensor() {
        delete data_;
        delete grad_;
    }
    
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
        cout << "data" << endl;
        data_->print();
        cout << "shape" << endl;
        cout << "(";
        for (int i = 0; i < data_->getDim(); i++) { 
            cout << data_->getShape()[i] << ", ";
        }
        cout << ")" << endl;
        cout << "grad" << endl;
        grad_->print();
    }

    void tensorAddScalar(T scalar) {
        for (int i = 0; i < data_->getSize(); i++) {
            data_->setData(i, data_->getData()[i] + scalar);
        }

        for (int i = 0; i < grad_->getSize(); i++) {
            grad_->setData(i, grad_->getData()[i] + 1);
        }
    }

    void tensorSubScalar(T scalar) {
        for (int i = 0; i < data_->getSize(); i++) {
            data_->setData(i, data_->getData()[i] - scalar);
        }

        for (int i = 0; i < grad_->getSize(); i++) {
            grad_->setData(i, grad_->getData()[i] - 1);
        }
    }

    void tensorMultScalar(T scalar) {
        
        for (int i = 0; i < data_->getSize(); i++) {
            data_->setData(i, data_->getData()[i] * scalar);
        }

        for (int i = 0; i < grad_->getSize(); i++) {
            grad_->setData(i, grad_->getData()[i] * scalar);
        }
    }


    void tensorDivisionScalar(T scalar) {
        
        for (int i = 0; i < data_->getSize(); i++) {
            data_->setData(i, data_->getData()[i] / scalar);
        }

        for (int i = 0; i < grad_->getSize(); i++) {
            grad_->setData(i, grad_->getData()[i] / scalar);
        }
    }
    
    // // requires matrix multplication
    // T matmul(T arr1[][], T arr2[][])
    // {
    //     for (i = 0; i < arr.size(); i++)
    //     {
    //         for (j = 0; j < arr.size(); j++)
    //         {
    //             arr1[i][j] = arr1[i][j] * arr2[i][j];
    //         }
    //     }
    // }
    // element wise multiplication
};
