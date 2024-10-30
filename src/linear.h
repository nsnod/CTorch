#pragma once
#include "linear.h"
#include "tensor.h"
#include <iostream>
#include <vector>

using namespace std;

class LinearLayer{
public:
    Tensor<float> weights;

    LinearLayer(int input_size, int output_size){
        weights = Tensor<float>(input_size, output_size); 
    }
    ~LinearLayer(){

    }

    void forward(Tensor<float> input){
        return input.matmul(weights);
    }
};