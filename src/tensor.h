#pragma once

#include "array.h"
#include <pthread.h>
#include <thread>

template <typename T = float>

class Tensor { 
 public:
    Array<T>* data_;
    Array<T>* grad_;
    Array<T>* prev_; // for storing the previous values of the tensor before an operation 
    vector<int> shape_;

    Tensor() : shape_({}), data_(nullptr), grad_(nullptr), prev_(nullptr) {}

    Tensor(vector<int> shape) : shape_(shape), data_(nullptr), grad_(nullptr), prev_(nullptr) {
        if (shape.size() == 1) {
            shape.push_back(1);  
        } 
        shape_ = shape;
        data_ = new Array<T>(shape_);
        grad_ = new Array<T>(shape_);
        prev_ = new Array<T>({3});    // default as 3 for now // should contain the previous values of the tensor 
    }

    Tensor(const Tensor& other) {
        shape_ = other.shape_;
        data_ = new Array<T>(*other.data_);
        grad_ = new Array<T>(*other.grad_);
        prev_ = new Array<T>(*other.prev_);
    }

    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            delete data_;
            delete grad_;
            shape_ = other.shape_;
            data_ = new Array<T>(*other.data_);
            grad_ = new Array<T>(*other.grad_);
            prev_ = new Array<T>(*other.prev_);
        }
        return *this;
    }

    ~Tensor() {
        delete data_;
        delete grad_;
        delete prev_;
    }

    // Functions 

    void reShape(vector<int> shape) {
        // if the shape isnt set yet
        if(shape_.size() == 0){
            for(auto dim : shape){
                shape_.push_back(dim);
            }
            // create new arrays with 0 values because tey didnt exist before 
            delete data_;
            delete grad_;
            data_ = new Array<T>(shape_);
            grad_ = new Array<T>(shape_);
        } else {
            int shapeCheck = 1;
            for (int dim : shape) {
                shapeCheck *= dim;
            }
            if (shapeCheck != data_->size_) {
                cout << "You cannot reshape to those dimensions. Try again." << endl;
                exit(EXIT_FAILURE);
            } 
        }

        data_->shape_ = shape;
        grad_->shape_ = shape;
        data_->calcStrides();
        grad_->calcStrides();
        data_->dimension_ = shape.size();
        grad_->dimension_ = shape.size();

    }

    //no need for tensor zero. the array is inhertly 0.
    void randomize_tensor(float lower, float upper) {
        // check if the arrays are initialized properly
        if (data_ != nullptr && grad_ != nullptr) {
            data_->randomize(lower, upper);
            grad_->randomize(lower, upper);
        } else {
            cout << "Error: Tensor arrays not properly initialized!" << endl;
            exit(EXIT_FAILURE);
        }
    }

    void print_tensor() {
        if (shape_.size() == 0 || (data_ == nullptr && grad_ == nullptr)) {
            cout << "Tensor is empty or uninitialized." << endl;
            return;
        }   
        cout << "Tensor:" << endl;
        cout << "shape" << endl;
        cout << "(";
        for (int i = 0; i < data_->dimension_; i++) { 
            cout << data_->shape_[i] << ", ";
        }
        cout << ")" << endl << endl;
        cout << "data" << endl;
        data_->print();
        cout << endl;
        cout << "grad" << endl;
        if (grad_ == nullptr) {
            cout << "Gradient has not been set for this tensor yet." << endl;
        } else {
            grad_->print();
        }
        cout << endl;
    }

    Tensor<T>& operator+=(T scalar) {
        int numThreads = 16;
        int chunkSize = (data_->size_ + numThreads - 1) / numThreads;
        std::vector<std::thread> threads(numThreads * 2);

        prev_ = new Array<T>(this->data_);

        for (int t = 0; t < numThreads; ++t) {
            int startIdx = t * chunkSize;
            int endIdx = std::min(startIdx + chunkSize, data_->size_);

            threads[t] = std::thread([this, startIdx, endIdx, scalar]() {
                for (int i = startIdx; i < endIdx; ++i) {
                    this->data_->data_[i] = this->data_->data_[i] + scalar;
                }
            });
        }

        for (int t = 0; t < numThreads; ++t) {
            int startIdx = t * chunkSize;
            int endIdx = std::min(startIdx + chunkSize, grad_->size_);

            threads[numThreads + t] = std::thread([this, startIdx, endIdx, scalar]() {
                for (int i = startIdx; i < endIdx; ++i) {
                    this->grad_->data_[i] = this->grad_->data_[i] + 1;
                }
            });
        }

        for (auto& th : threads) {
            if (th.joinable()) {
                th.join();
            }
        }

        *output_ = 

        return *this;
    }

    Tensor<T>& operator-=(T scalar) {
        for (int i = 0; i < data_->size_; i++) {
            data_->data_[i] = data_->data_[i] - scalar;
        }

        for (int i = 0; i < grad_->size_; i++) {
            grad_->data_[i] = grad_->data_[i] - 1;
        }

        return *this;
    }

    Tensor<T>& operator*=(T scalar) {
        
        for (int i = 0; i < data_->size_; i++) {
            data_->data_[i] = data_->data_[i] * scalar;
        }

        for (int i = 0; i < grad_->size_; i++) {
            grad_->data_[i] = grad_->data_[i] * scalar;
        }

        return *this;
    }


    Tensor<T>& operator/=(T scalar) {
        
        for (int i = 0; i < data_->size_; i++) {
            data_->data_[i] = data_->data_[i] / scalar;
        }

        for (int i = 0; i < grad_->size_; i++) {
            grad_->data_[i] = grad_->data_[i] / scalar;
        }

        return *this;
    }


    // ONLY WORKS FOR 1D AND 2D CURRENTLY
    // in place until we expand for CNN 3ds
    Tensor operator*(const Tensor& other) const {
        Array<T>* data = this->data_;
        Array<T>* multData = other.data_;

        vector<int> dataShape = data_->shape_;
        vector<int> multShape = other.data_->shape_;
        vector<int> outputShape;

        if (dataShape.size() != 2 || multShape.size() != 2) {
            cout << "Error: Multiplication only supports 2D tensors!" << endl;
            exit(EXIT_FAILURE);
        }

        if (dataShape[1] != multShape[0]) {
            cout << "Error: Incompatible shapes for multiplication!" << endl;
            cout << "Shape of tensor A: (" << dataShape[0] << ", " << dataShape[1] << ")" << endl;
            cout << "Shape of tensor B: (" << multShape[0] << ", " << multShape[1] << ")" << endl;
            exit(EXIT_FAILURE);
        }

        outputShape.push_back(dataShape[0]);
        outputShape.push_back(multShape[1]); 

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

        Tensor<T> result;
        result.shape_ = outputShape;
        result.data_ = output;
        result.grad_ = nullptr;

        return result;
    }


    T operator[] (vector<int> indicies) {
        if (indicies.size() != shape_.size()) {
            cout << endl;
            cout << "This is not a valid index!" << endl;
            exit(EXIT_FAILURE);
        } 

        for (int i = 0; i < indicies.size(); i++) {
            if (indicies[i] >= shape_[i]) {
                cout << endl;
                cout << "This is not a valid index!" << endl;
                exit(EXIT_FAILURE);
            }
        }

        return data_->at(indicies);

    }
};