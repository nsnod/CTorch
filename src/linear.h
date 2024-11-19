#pragma once
#include <iostream>
#include "../src/tensor.h"

using namespace std;

class LinearLayer{
public:

    Tensor<float> weights;
    

    LinearLayer(int input_size, int output_size){ 
        // Initialize weights with the shape (output_size, input_size) for matrix multiplication
        weights.setShape({input_size, output_size});

    }

    Tensor <double> forward(const Tensor<float>& X) { 
        //Print the shape of X
        //X.print_tensor();
        // Multiply weights with X

       int row = (weights.getShape())[0];

        Tensor<float> output(,)

        
    }
};