#pragma once
#include <iostream>
#include "../src/tensor.h"

using namespace std;

class LinearLayer{
public:
    Tensor<float> weights;
    
    LinearLayer(const int input_size, const int output_size){ 
        // Initialize weights with the shape (output_size, input_size) for matrix multiplication
        weights.setShape({input_size, output_size});
        weights.randomize_tensor(-1, 1);
    }

    Tensor <float> forward(Tensor<float>& X) { 
        // Multiply weights with X
        Tensor<float> output_tensor({(weights.getShape()).at(0),(X.getShape()).at(1)});//setting dimensions (nxm) (x*y) = n*y matrix//
        output_tensor.randomize_tensor(-1, 1);
        output_tensor.tensorMulData(output_tensor.getData(), X.getData());

        return output_tensor;
    }
};