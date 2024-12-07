#include "../src/tensor.h"

template <typename T = float>
Tensor<T> relu(Tensor<T>* input){
    Tensor<T>* t = new Tensor<T>(input->shape_);
    // update previous 
    input->prev_->push_back(input->data_);
    for(int i = 0; i < input->data_->size_; i++){
        for(int j = 0; j < input->data_->size_; j++){
            t->data_[i][j] = input->data_[i][j] > 0 ? input->data_[i][j] : 0;
        }
    }
    return *t;
}

template <typename T = float>
Tensor<T> softmax(Tensor<T>* inp){
    if (inp->shape_.size() != 2) {
        std::cout << "Must be a 2D tensor" << std::endl;
        exit(EXIT_FAILURE);
    }

    int batch_size = inp->shape_[0];
    int num_classes = inp->shape_[1];
    Tensor<T>* t = new Tensor<T>(inp->shape_);
    for (int i = 0; i < batch_size; i++) {
        // Find the max value in the row for numerical stability
        T maxVal = inp->data_->at({i, 0});
        for (int j = 1; j < num_classes; j++) {
            T val = inp->data_->at({i, j});
            if (val > maxVal) {
                maxVal = val;
            }
        }

        // Compute exponentials and sum
        std::vector<T> exp_values(num_classes);
        T sum_exp = 0;
        for (int j = 0; j < num_classes; j++) {
            T exp_val = std::exp(inp->data_->at({i, j}) - maxVal);
            exp_values[j] = exp_val;
            sum_exp += exp_val;
        }

        // Compute softmax
        for (int j = 0; j < num_classes; j++) {
            t->data_->at({i, j}) = exp_values[j] / sum_exp;
        }
    }
    return *t;
}