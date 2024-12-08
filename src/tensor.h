#pragma once
#include <mpi.h>
#include <vector>
#include "array.h"
#include <pthread.h>
#include <thread>

template <typename T = float>

class Tensor { 
 public:
    Array<T>* data_;
    Array<T>* grad_;
    vector<int> shape_;

    Tensor() : shape_({}), data_(nullptr), grad_(nullptr) {}

    Tensor(vector<int> shape) : shape_(shape), data_(nullptr), grad_(nullptr) {
        if (shape.size() == 1) {
            shape.push_back(1);  
        } 
        shape_ = shape;
        data_ = new Array<T>(shape_);
        grad_ = new Array<T>(shape_);
    }

    Tensor(const Tensor& other) {
        shape_ = other.shape_;
        data_ = new Array<T>(*other.data_);
        grad_ = new Array<T>(*other.grad_);
    }

    Tensor& operator=(const Tensor& other) {
        if (this != &other) {
            delete data_;
            delete grad_;
            shape_ = other.shape_;
            data_ = new Array<T>(*other.data_);
            grad_ = new Array<T>(*other.grad_);
        }
        return *this;
    }

    ~Tensor() {
        delete data_;
        delete grad_;
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

    Tensor<T>& tensorAdd(T scalar) {
        for (int i = 0; i < data_->size_; i++) {
            data_->data_[i] = data_->data_[i] + scalar;
        }

        for (int i = 0; i < grad_->size_; i++) {
            grad_->data_[i] = grad_->data_[i] + 1;
        }

        return *this;
    }

    Tensor<T>& operator+(T scalar){
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        int rowsPerProcess = data_->shape_[0] / size;
        int extraRows = data_->shape_[0] % size;
        int startRow = rank * rowsPerProcess + min(rank, extraRows);
        int endRow = startRow + rowsPerProcess + (rank < extraRows ? 1 : 0);

        for (int i = startRow; i < endRow; ++i) {
            for (int j = 0; j < data_->shape_[1]; ++j) {
                vector<int> index = {i, j};
                data_->at(index) += scalar;
            }
        }

        MPI_Allreduce(MPI_IN_PLACE, data_->data_.data(), data_->size_, 
                    std::is_same<T, float>::value ? MPI_FLOAT : MPI_DOUBLE,
                    MPI_SUM, MPI_COMM_WORLD);

        return *this;
    }


    Tensor<T>& tensorSub(T scalar) {
        for (int i = 0; i < data_->size_; i++) {
            data_->data_[i] = data_->data_[i] - scalar;
        }

        for (int i = 0; i < grad_->size_; i++) {
            grad_->data_[i] = grad_->data_[i] - 1;
        }

        return *this;
    }

    Tensor<T>& operator-(T scalar) {
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        int rowsPerProcess = data_->shape_[0] / size;
        int extraRows = data_->shape_[0] % size;
        int startRow = rank * rowsPerProcess + min(rank, extraRows);
        int endRow = startRow + rowsPerProcess + (rank < extraRows ? 1 : 0);

        for (int i = startRow; i < endRow; ++i) {
            for (int j = 0; j < data_->shape_[1]; ++j) {
                vector<int> index = {i, j};
                data_->at(index) -= scalar;
            }
        }

        MPI_Allreduce(MPI_IN_PLACE, data_->data_.data(), data_->size_, 
                    std::is_same<T, float>::value ? MPI_FLOAT : MPI_DOUBLE,
                    MPI_SUM, MPI_COMM_WORLD);

        return *this;
    }

  

    Tensor<T>& tensorScalarMult(T scalar) {
        for (int i = 0; i < data_->size_; i++) {
            data_->data_[i] = data_->data_[i] * scalar;
        }

        for (int i = 0; i < grad_->size_; i++) {
            grad_->data_[i] = grad_->data_[i] * scalar;
        }

        return *this;
    }

    Tensor<T>& operator*(T scalar) {
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        int rowsPerProcess = data_->shape_[0] / size;
        int extraRows = data_->shape_[0] % size;
        int startRow = rank * rowsPerProcess + min(rank, extraRows);
        int endRow = startRow + rowsPerProcess + (rank < extraRows ? 1 : 0);

        for (int i = startRow; i < endRow; ++i) {
            for (int j = 0; j < data_->shape_[1]; ++j) {
                vector<int> index = {i, j};
                data_->at(index) *= scalar;
            }
        }

        MPI_Allreduce(MPI_IN_PLACE, data_->data_.data(), data_->size_, 
                    std::is_same<T, float>::value ? MPI_FLOAT : MPI_DOUBLE,
                    MPI_SUM, MPI_COMM_WORLD);

        return *this;
    }

    Tensor<T>& tensorScalarDivide(T scalar) {
        
        for (int i = 0; i < data_->size_; i++) {
            data_->data_[i] = data_->data_[i] / scalar;
        }

        for (int i = 0; i < grad_->size_; i++) {
            grad_->data_[i] = grad_->data_[i] / scalar;
        }

        return *this;
    }

    Tensor<T>& operator/(T scalar) {
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        int rowsPerProcess = data_->shape_[0] / size;
        int extraRows = data_->shape_[0] % size;
        int startRow = rank * rowsPerProcess + min(rank, extraRows);
        int endRow = startRow + rowsPerProcess + (rank < extraRows ? 1 : 0);

        for (int i = startRow; i < endRow; ++i) {
            for (int j = 0; j < data_->shape_[1]; ++j) {
                vector<int> index = {i, j};
                data_->at(index) /= scalar;
            }
        }

        MPI_Allreduce(MPI_IN_PLACE, data_->data_.data(), data_->size_, 
                    std::is_same<T, float>::value ? MPI_FLOAT : MPI_DOUBLE,
                    MPI_SUM, MPI_COMM_WORLD);

        return *this;
    }
    

    // ONLY WORKS FOR 1D AND 2D CURRENTLY
    // in place until we expand for CNN 3ds
    Tensor operator*(const Tensor& other) const {
        Array<T>* data = this->data_;
        Array<T>* multData = other.data_;

        vector<int> dataShape = data_->shape_;
        vector<int> multShape = multData->shape_;
        vector<int> outputShape;

        
        if (dataShape.size() != 2 || multShape.size() != 2) {
            cout << "Error: Multiplication only supports 2D tensors!" << endl;
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        if (dataShape[1] != multShape[0]) {
            cout << "Error: Incompatible shapes for multiplication!" << endl;
            cout << "Shape of tensor A: (" << dataShape[0] << ", " << dataShape[1] << ")" << endl;
            cout << "Shape of tensor B: (" << multShape[0] << ", " << multShape[1] << ")" << endl;
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        outputShape.push_back(dataShape[0]);
        outputShape.push_back(multShape[1]);

        Array<T>* output = new Array<T>(outputShape);

        // MPI CODE
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        int rowsPerProcess = dataShape[0] / size;
        int extraRows = dataShape[0] % size;

        int startRow = rank * rowsPerProcess + min(rank, extraRows);
        int endRow = startRow + rowsPerProcess + (rank < extraRows ? 1 : 0);

        for (int i = startRow; i < endRow; i++) {
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

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(
            MPI_IN_PLACE,
            output->data_.data(), 
            output->size_,        
            std::is_same<T, float>::value ? MPI_FLOAT : MPI_DOUBLE, // ensures it works for float and doubles. This might need to be changed later
            MPI_SUM,
            MPI_COMM_WORLD
        );

        Tensor<T> result;
        result.shape_ = outputShape;
        result.data_ = output;
        result.grad_ = nullptr;

        return result;
    }

    Tensor<T> non_parallel_tensor_mult_test(const Tensor<T>& other) const {
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