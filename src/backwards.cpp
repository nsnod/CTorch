#include "../src/tensor.h"

// Gradient Calculations 

template <typename T = float>
void relu_backward(Tensor<T>* output) {
    for(int i = 0; i < output->data_.size(); i++){
        output->grad_[i] = *(output->data_)[i] > 0 ? *(output->data_)[i] : 0;
    }
}


template <typename T = float>
void softmax_backward(Tensor<T>* inp, Tensor<T>* out) {
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
            T grad_sj = out->grad_->at({i, j});
            T sj = out->data_->at({i, j});
            dot += grad_sj * sj;
        }

        // Compute gradient with respect to the input
        for (int k = 0; k < num_classes; k++) {
            T sk = out->data_->at({i, k});
            T grad_sk = out->grad_->at({i, k});
            inp->grad_->at({i, k}) = sk * (grad_sk - dot);
        }
    }
}