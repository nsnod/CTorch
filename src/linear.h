#pragma once
#include <iostream>
#include <vector>
#include "../src/tensor.h"

using namespace std;

class LinearLayer {
public:
    Eigen::MatrixXd weights;

    LinearLayer(int input_size, int output_size) {
        // Initialize weights with the shape (output_size, input_size) for matrix multiplication
        weights = Eigen::MatrixXd::Random(output_size, input_size);
    }

    Eigen::VectorXd forward(const Eigen::VectorXd& X) { 
        // Print the shape of X
        cout << "X shape: " << X.rows() << " " << X.cols() << endl;

        // Multiply weights with X
        return weights * X;
    }
};