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
    /*
        TODO prev_ NEEDS TO BE 3D TO CONTAIN LIST OF PREV RESULTS
        every time an oepration is done the previous reuslt should be stored in the correct index
    */
    vector<Array<T>*>* prev_; // for storing the previous values of the tensor before an operation 
    vector<int> shape_;
    string operation_;  // contains the operation that was done to create this tensor

    Tensor() : shape_({}), data_(nullptr), grad_(nullptr), prev_(nullptr), operation_(""){}

    Tensor(vector<int> shape) : shape_(shape), data_(nullptr), grad_(nullptr), prev_(nullptr), operation_("") {
        if (shape.size() == 1) {
            shape.push_back(1);  
        } 
        shape_ = shape;
        data_ = new Array<T>(shape_);
        grad_ = new Array<T>(shape_);
        prev_ = new vector<Array<T>*>(3, nullptr); // default allocate 3 // should contain the previous values used to create the tensor if any
        operation_ = "";
    }

    Tensor(const Tensor& other) {
        shape_ = other.shape_;
        data_ = new Array<T>(*other.data_);
        grad_ = new Array<T>(*other.grad_);
        prev_ = other.prev_;    // TODO if we want to delete this itll get messay since the other tensor will delete it too
        operation_ = other.operation_;
    }

    Tensor* operator=(const Tensor* other) {
        if (this != &other) {
            delete data_;
            delete grad_;
            shape_ = other->shape_;
            data_ = new Array<T>(*other->data_);
            grad_ = new Array<T>(*other->grad_);
            prev_ = other->prev_;    // TODO if we want to delete this itll get messay since the other tensor will delete it too
            operation_ = other->operation_;
        }
        return this;
    }

    ~Tensor() {
        delete data_;
        delete grad_;
        delete prev_;
    }

    // Functions

    void resetZeroData() const {
        if (data_ == nullptr) {
            std::cout << "No gradient to reset..." << std::endl;
            return;
        }
        std::fill(data_->data_.begin(), data_->data_.end(), static_cast<T>(0.0f));
    }

    void resetZeroGrad() {
        if (grad_ == nullptr) {
            std::cout << "No gradient to reset..." << std::endl;
            return;
        }
        std::fill(grad_->data_.begin(), grad_->data_.end(), static_cast<T>(0.0f));
    }

    void reshape(vector<int> shape) {
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
        cout << "prev" << endl;
        if (prev_ == nullptr && prev_->at(0) == nullptr) {
            cout << "Prev has not been set for this tensor yet." << endl;
        } else {
            for(int i = 0; i < prev_->size(); i++){
                if (prev_->at(i) == nullptr) {
                    cout << "Prev has not been set for this tensor yet." << endl;
                } else {
                    prev_->at(i)->print();
                }
            }
        }
        cout << endl;
    }


    /*
        ======================
        TENSOR AND SCALAR OPS
        ======================
    */

    Tensor<T>& operator+=(T scalar) {
        int numThreads = 16;

        int chunkSize = (data_->size_ + numThreads - 1) / numThreads;
        std::vector<std::thread> threads(numThreads * 2);

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
                grad_->at(index) += 1;
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, data_->data_.data(), data_->size_, 
                    std::is_same<T, float>::value ? MPI_FLOAT : MPI_DOUBLE,
                    MPI_SUM, MPI_COMM_WORLD);

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
                grad_->at(index) -= 1;
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, data_->data_.data(), data_->size_, 
                    std::is_same<T, float>::value ? MPI_FLOAT : MPI_DOUBLE,
                    MPI_SUM, MPI_COMM_WORLD);

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
                grad_->at(index) *= scalar;
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, data_->data_.data(), data_->size_, 
                    std::is_same<T, float>::value ? MPI_FLOAT : MPI_DOUBLE,
                    MPI_SUM, MPI_COMM_WORLD);

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
                grad_->at(index) /= scalar;
                
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(MPI_IN_PLACE, data_->data_.data(), data_->size_, 
                    std::is_same<T, float>::value ? MPI_FLOAT : MPI_DOUBLE,
                    MPI_SUM, MPI_COMM_WORLD);

        return *this;
    }



    /*
        ======================
        TENSOR AND TENSOR OPS
        ======================
    */

    // ONLY WORKS FOR 1D AND 2D CURRENTLY
    // in place until we expand for CNN 3ds
    Tensor operator*(const Tensor& other) const {
        Array<T>* data = this->data_;
        Array<T>* multData = other.data_;
        Array<T>* grad = this->grad_;
        Array<T>* multGrad = other.grad_;

        vector<int> dataShape = data_->shape_;
        vector<int> multShape = multData->shape_;

        
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

        (*prev_)[0] = this->data_;
        (*prev_)[1] = other.data_;
        vector<int> outputShape = {dataShape[0], multShape[1]};
        Array<T>* output = new Array<T>(outputShape);

        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        int rowsPerProcess = dataShape[0] / size;
        int extraRows = dataShape[0] % size;

        int startRow = rank * rowsPerProcess + min(rank, extraRows);
        int endRow = startRow + rowsPerProcess + (rank < extraRows ? 1 : 0);

        for (int i = startRow; i < endRow; i++) {
            vector<T> rowBuffer(multShape[1], 0);
            for (int j = 0; j < multShape[1]; j++) {
                T sum = 0;
                for (int k = 0; k < dataShape[1]; k++) {
                    vector<int> indexA = {i, k};
                    vector<int> indexB = {k, j};
                    sum += data->at(indexA) * multData->at(indexB);
                }
                rowBuffer[j] = sum;
            }
            for (int j = 0; j < multShape[1]; j++) {
                vector<int> indexOutput = {i, j};
                output->at(indexOutput) = rowBuffer[j];
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(
            MPI_IN_PLACE,
            data->data_.data(), 
            data->size_,        
            std::is_same<T, float>::value ? MPI_FLOAT : MPI_DOUBLE, // ensures it works for float and doubles. This might need to be changed later
            MPI_SUM,
            MPI_COMM_WORLD
        );

        Tensor<T> result;
        result.shape_ = outputShape;
        result.data_ = output;
        result.grad_ = nullptr;
        result.operation_ = "matmul";

        return result;
    }

    // ONLY FOR TESTING PURPOSES REMOVE LATER 
    Tensor<T> non_parallel_tensor_mult_test(const Tensor<T>& other) const {
        Array<T>* data = this->data_;
        Array<T>* multData = other.data_;
        Array<T>* grad = this->grad_;
        Array<T>* multGrad = other.grad_;

        vector<int> dataShape = data_->shape_;
        vector<int> multShape = multData->shape_;

        
        if (dataShape.size() != 2 || multShape.size() != 2) {
            cout << "Error: Multiplication only supports 2D tensors!" << endl;
        }

        if (dataShape[1] != multShape[0]) {
            cout << "Error: Incompatible shapes for multiplication!" << endl;
            cout << "Shape of tensor A: (" << dataShape[0] << ", " << dataShape[1] << ")" << endl;
            cout << "Shape of tensor B: (" << multShape[0] << ", " << multShape[1] << ")" << endl;
        }

        vector<int> outputShape = {dataShape[0], multShape[1]};
        Array<T>* output = new Array<T>(outputShape);

        for (int i = 0; i < dataShape[0]; i++) {
            vector<T> rowBuffer(multShape[1], 0);
            for (int j = 0; j < multShape[1]; j++) {
                T sum = 0;
                T gradientSum = 0;
                for (int k = 0; k < dataShape[1]; k++) {
                    vector<int> indexA = {i, k};
                    vector<int> indexB = {k, j};
                    sum += data->at(indexA) * multData->at(indexB);
                }
                rowBuffer[j] = sum;
            }
            for (int j = 0; j < multShape[1]; j++) {
                vector<int> indexOutput = {i, j};
                output->at(indexOutput) = rowBuffer[j];
            }
        }

        Tensor<T> result;
        result.shape_ = outputShape;
        result.data_ = output;
        result.grad_ = nullptr;

        return result;
    }

    // Tensor addition 
    Tensor<T>* operator+(Tensor& other) {
        // check if the shapes are the same
        if (shape_ != other.shape_) {
            cout << "Error: Incompatible shapes for addition!" << endl;
            cout << "Shape of tensor A: ";
            for (int i = 0; i < shape_.size(); i++) {
                cout << shape_[i] << " ";
            }
            cout << endl;
            cout << "Shape of tensor B: ";
            for (int i = 0; i < other.shape_.size(); i++) {
                cout << other.shape_[i] << " ";
            }
            cout << endl;
            exit(EXIT_FAILURE);
        }

        int numThreads = 16;
        int chunkSize = (data_->size_ + numThreads - 1) / numThreads;
        vector<thread> threads(numThreads * 2);
        Tensor<T>* output = new Tensor<T>(shape_);

        (*prev_)[0] = this->data_;
        (*prev_)[1] = other.data_;
        output->operation_ = "add";

        for (int t = 0; t < numThreads; ++t) {
            int startIdx = t * chunkSize;
            int endIdx = std::min(startIdx + chunkSize, data_->size_);

            threads[t] = std::thread([this, startIdx, endIdx, other]() {
                for (int i = startIdx; i < endIdx; ++i) {
                    output->data_->data_[i] = this->data_->data_[i] + other.data_->data_[i];
                }
            });
        }

        for (auto& th : threads) {
            if (th.joinable()) {
                th.join();
            }
        }

        return output;
    }

    // Tensor addition 
    Tensor<T>* operator-(Tensor& other) {
        // check if the shapes are the same
        if (shape_ != other.shape_) {
            cout << "Error: Incompatible shapes for addition!" << endl;
            cout << "Shape of tensor A: ";
            for (int i = 0; i < shape_.size(); i++) {
                cout << shape_[i] << " ";
            }
            cout << endl;
            cout << "Shape of tensor B: ";
            for (int i = 0; i < other.shape_.size(); i++) {
                cout << other.shape_[i] << " ";
            }
            cout << endl;
            exit(EXIT_FAILURE);
        }

        int numThreads = 16;
        int chunkSize = (data_->size_ + numThreads - 1) / numThreads;
        vector<thread> threads(numThreads * 2);
        Tensor<T>* output = new Tensor<T>(shape_);

        (*prev_)[0] = this->data_;
        (*prev_)[1] = other.data_;

        for (int t = 0; t < numThreads; ++t) {
            int startIdx = t * chunkSize;
            int endIdx = std::min(startIdx + chunkSize, data_->size_);

            threads[t] = std::thread([this, startIdx, endIdx, other]() {
                for (int i = startIdx; i < endIdx; ++i) {
                    output->data_->data_[i] = this->data_->data_[i] - other.data_->data_[i];
                }
            });
        }

        for (auto& th : threads) {
            if (th.joinable()) {
                th.join();
            }
        }

        return output;
    }

    // Tensor addition 
    Tensor<T>& operator+=(Tensor& other) {
        // check if the shapes are the same
        if (shape_ != other.shape_) {
            cout << "Error: Incompatible shapes for addition!" << endl;
            cout << "Shape of tensor A: ";
            for (int i = 0; i < shape_.size(); i++) {
                cout << shape_[i] << " ";
            }
            cout << endl;
            cout << "Shape of tensor B: ";
            for (int i = 0; i < other.shape_.size(); i++) {
                cout << other.shape_[i] << " ";
            }
            cout << endl;
            exit(EXIT_FAILURE);
        }

        int numThreads = 16;
        int chunkSize = (data_->size_ + numThreads - 1) / numThreads;
        std::vector<std::thread> threads(numThreads * 2);

        (*prev_)[0] = this->data_;
        (*prev_)[1] = other.data_;

        for (int t = 0; t < numThreads; ++t) {
            int startIdx = t * chunkSize;
            int endIdx = std::min(startIdx + chunkSize, data_->size_);

            threads[t] = std::thread([this, startIdx, endIdx, other]() {
                for (int i = startIdx; i < endIdx; ++i) {
                    this->data_->data_[i] = this->data_->data_[i] + other.data_->data_[i];
                }
            });
        }

        for (auto& th : threads) {
            if (th.joinable()) {
                th.join();
            }
        }

        return *this;
    }

    // Tensor subtraction 
    Tensor<T>& operator-=(Tensor& other) {
        // check if the shapes are the same
        if (shape_ != other.shape_) {
            cout << "Error: Incompatible shapes for addition!" << endl;
            cout << "Shape of tensor A: ";
            for (int i = 0; i < shape_.size(); i++) {
                cout << shape_[i] << " ";
            }
            cout << endl;
            cout << "Shape of tensor B: ";
            for (int i = 0; i < other.shape_.size(); i++) {
                cout << other.shape_[i] << " ";
            }
            cout << endl;
            exit(EXIT_FAILURE);
        }

        int numThreads = 16;
        int chunkSize = (data_->size_ + numThreads - 1) / numThreads;
        std::vector<std::thread> threads(numThreads * 2);

        (*prev_)[0] = this->data_;
        (*prev_)[1] = other.data_;

        for (int t = 0; t < numThreads; ++t) {
            int startIdx = t * chunkSize;
            int endIdx = std::min(startIdx + chunkSize, data_->size_);

            threads[t] = std::thread([this, startIdx, endIdx, other]() {
                for (int i = startIdx; i < endIdx; ++i) {
                    this->data_->data_[i] = this->data_->data_[i] - other.data_->data_[i];
                }
            });
        }

        for (auto& th : threads) {
            if (th.joinable()) {
                th.join();
            }
        }

        Tensor<T> result;
        result.shape_ = shape_;
        result.data_ = result;
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

    // used for testing 
    void clear_prev(){
        if (prev_ != nullptr && prev_->at(0) != nullptr){
            for(int i = 0; i < prev_->size(); i++){
                prev_->at(i) = nullptr;
            }            
        }
    }

};