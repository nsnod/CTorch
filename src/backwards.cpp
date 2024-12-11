#include "../src/tensor.h"

// Backwards
template<typename T = float>
void backward(Tensor<T>* input){
    if(input->operation_ == "relu"){
        relu_backward(input);
    } else if(input->operation_ == "softmax"){
        softmax_backward(input);
    } else if(input->operation_ == "mul"){
        mul_backward(input);
    } else {
        cout << "Operation not supported" << endl;
        exit(EXIT_FAILURE);
    }
    for(int i = 0; i < input->num_prev; i++){
        backward(input->prev_->at(i));  // recursively performs backprop on each node
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

    // Initialize grad_ of inp if not already initialized
    if (inp->grad_ == nullptr) {
        inp->grad_ = new Array<T>(inp->shape_);
    }
    
    for (int i = 0; i < batch_size; i++) {
        // Compute dot product of grad_s and s
        T dot = 0;
        for (int j = 0; j < num_classes; j++) {
            T grad_sj = (*inp->prev_)[0]->grad_->at({i, j});
            T sj = (*inp->prev_)[0]->data_->at({i, j});
            dot += grad_sj * sj;
        }

        // Compute gradient with respect to the input
        for (int k = 0; k < num_classes; k++) {
            T sk = (*inp->prev_)[0]->data_->at({i, k});
            T grad_sk = (*inp->prev_)[0]->grad_->at({i, k});
            inp->grad_->at({i, k}) = sk * (grad_sk - dot);
        }
    }
}