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

    Tensor <float> forward(Tensor<float>& X) { 
        // Multiply weights with X
        Tensor<float> output_tensor({(weights.shape_).at(0),(X.shape_).at(1)});//setting dimensions (nxm) (x*y) = n*y matrix//
        output_tensor = output_tensor * X;
        return output_tensor;
    }
};