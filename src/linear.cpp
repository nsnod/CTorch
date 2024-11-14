
#include <iostream>
#include "linear.h"
#include <Eigen/Dense>

using namespace std;

int main() {
    LinearLayer linear = LinearLayer(784, 10);
    Eigen::VectorXd vec = Eigen::VectorXd::Random(784, 1);

    std::cout << "Result:\n" << linear.forward(vec) << std::endl;

    return 0;
}
