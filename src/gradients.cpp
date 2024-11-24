#include "../src/tensor.h"

template <typename T = float>

// Gradient Calculations 
void relu_backward(Tensor<T>* output) {
    for(int i = 0; i < output.size(); i++){
        output->grad_[i] = *(output->data_)[i] > 0 ? *(output->data_)[i] : 0;
    }
}