#include <iostream>
#include "tensor.h"

int main() {
    Tensor<float> matA({2, 2});
    Tensor<float> matB({2, 2});
    Tensor<float> matC({2, 2});
    Tensor<float> mulMe({1, 3});

    // Initialize matrices
    matA.randomize_tensor(0, 1);
    matB.randomize_tensor(0, 1);
    matC.randomize_tensor(0, 1);

    // Perform matrix multiplication
    matC.tensorMulData(matA.getData(), matB.getData());

    matC.print_tensor();
    
    return 0;
}