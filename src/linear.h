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

    Tensor <float> forward(Tensor<float>& X) { 
        //Print the shape of X
        //X.print_tensor();
        // Multiply weights with X

        Tensor<float> output_tensor;

       output_tensor.setShape({(weights.getShape()).at(0),(X.getShape()).at(1)}); //setting dimensions (nxm) (x*y) = n*y matrix//

       output_tensor.tensorMulData(weights.getData(),X.getData());

        return output_tensor;

    }
};