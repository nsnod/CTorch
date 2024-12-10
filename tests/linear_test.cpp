#include <iostream>
#include "tensor.h"
#include "linear.h"

int main() {
    Tensor<float> matA({2, 2});
    // Tensor<float> matB({2, 2});
    // Tensor<float> matC({2, 2});
    // Tensor<float> mulMe({1, 3});

    // // Initialize matrices
    matA.randomize_tensor(0, 1);
    // matB.randomize_tensor(0, 1);
    // matC.randomize_tensor(0, 1);

    // // Perform matrix multiplication
    // matC.tensorMulData(matA.getData(), matB.getData());
    cout << "made it" << endl;


    LinearLayer test(2,2);

    Tensor<float> matB = test.forward(matA);
    matB.print_tensor();

    return 0;
}