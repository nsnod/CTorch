#include <Eigen/Dense>
#include <iostream>

int main() {
    Eigen::MatrixXd matA(2, 2);
    Eigen::MatrixXd matB(2, 2);

    // Initialize matrices
    matA << 1, 2,
            3, 4;
    matB << 5, 6,
            7, 8;

    // Perform matrix multiplication
    Eigen::MatrixXd result = matA * matB;

    std::cout << "Result:\n" << result << std::endl;

    return 0;
}