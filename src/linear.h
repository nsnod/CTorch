#pragma once
#include <iostream>
#include "../src/tensor.h"

using namespace std;

class LinearLayer{
public:
    Tensor<float> weights;
    
    LinearLayer(const int input_size, const int output_size){ 
        // Initialize weights with the shape (output_size, input_size) for matrix multiplication
        weights.reshape({input_size, output_size});
        weights.randomize_tensor(-1, 1);
    }

    Tensor<float> forward(Tensor<float>& X) { 
        Tensor<float> output_tensor({weights.shape_[0], X.shape_[1]});
        output_tensor = output_tensor * X; 
        return output_tensor;
    }
};