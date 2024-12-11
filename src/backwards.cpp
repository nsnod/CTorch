#include "../src/tensor.h"
#include <mpi.h>

// Backwards
template<typename T = float>
void backward(Tensor<T>* input){
    if(input->operation_ == "relu"){
        cout << "relu backward" << endl;
        relu_backward(input);
    } else if(input->operation_ == "softmax"){
        cout << "softmax backward" << endl;
        softmax_backward(input);
    } else if(input->operation_ == "mul"){
        cout << "mul backward" << endl;
        mul_backward(input);
    } else if(input->operation_ == "matmul"){
        cout << "matmul backward" << endl;
        matmul_backward(input);
    } else {
        cout << "Operation " << input->operation_ << " not supported" << endl;
    }
    cout << "num_prev: " << input->num_prev << endl;
    for(int i = 0; i < input->num_prev; i++){
        backward((*input->prev_)[i]);  // recursively performs backprop on each node
    }
}


// Gradient Calculations 

template <typename T = float>
void mul_backward(Tensor<T>* input) {
    for(int i = 0; i < input->shape_[0]; i++){
        for(int j = 0; j < input->shape_[1]; j++){
            (*input->prev_)[0]->grad_->at({i, j}) = (*input->prev_)[0]->data_->at({i, j}) * input->grad_->at({i, j});
            (*input->prev_)[1]->grad_->at({i, j}) = (*input->prev_)[1]->data_->at({i, j}) * input->grad_->at({i, j});
        }
    }
}

template <typename T = float>
void matmul_backward(Tensor<T>* input) {
    Array<T>* data = (*input->prev_)[0]->data_;
    Array<T>* multData = (*input->prev_)[1]->data_;
    
    vector<int> dataShape = data->shape_;
    vector<int> multShape = multData->shape_;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rowsPerProcess = multShape[0] / size;
    int extraRows = multShape[0] % size;
    int startRow = rank * rowsPerProcess + min(rank, extraRows);
    int endRow = startRow + rowsPerProcess + (rank < extraRows ? 1 : 0);
    for (int i = startRow; i < endRow; i++) {
        vector<T> rowBuffer(dataShape[1], 0);
        for (int j = 0; j < dataShape[1]; j++) {
            T tmp = 0;
            for (int k = 0; k < multShape[1]; k++) {
                vector<int> indexA = {i, k};
                vector<int> indexB = {k, j};
                tmp += input->grad_->at(indexA) * (*input->prev_)[1]->data_->at(indexB);
            }
            rowBuffer[j] = tmp;
        }
        for (int j = 0; j < dataShape[1]; j++) {
            vector<int> indexOutput = {i, j};
            cout << rowBuffer[j] << endl;
            (*input->prev_)[0]->grad_->at(indexOutput) = rowBuffer[j];
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
}

template <typename T = float>
void relu_backward(Tensor<T>* input) {
    for(int i = 0; i < input->shape_[0]; i++){
        for(int j = 0; j < input->shape_[1]; j++){
            (*input->prev_)[0]->grad_->at({i, j}) = (*input->prev_)[0]->data_->at({i, j}) > 0 ? input->grad_->at({i, j}) : 0;
        }
    }
}


template <typename T = float>
void softmax_backward(Tensor<T>* inp) {
    if (inp->shape_.size() != 2) {
        std::cout << "Must be a 2D tensor" << std::endl;
        exit(EXIT_FAILURE);
    }

    int batch_size = inp->shape_[0];
    int num_classes = inp->shape_[1];
    
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_classes; j++){
            // case 1
            if (i == j){
                (*inp->prev_)[0]->grad_->at({i, j}) = inp->data_->at({i, j}) * (1 - inp->data_->at({i, j}));
            }
            // case 2
            else {
                (*inp->prev_)[0]->grad_->at({i, j}) = -1 * inp->data_->at({i, j}) * inp->data_->at({i, j});
            }
        }
    }
}
