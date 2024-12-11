#include <gtest/gtest.h>
#include "../src/tensor.h"
#include "../src/functions.cpp"
#include "../src/backwards.cpp"

int test1() {
    std::cout << "test" << std::endl;
    return 1;
}

TEST(ExampleTest, OneIsOne) {
    int result = test1();
    EXPECT_EQ(result, 1) << "Expected result to be 1, but got " << result;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Tensor<float> input({8, 4});
    Tensor<float> input2({4, 8});

    input.randomize_tensor(-1.0, 1.0); // Randomize the tensor
    input2.randomize_tensor(-1.0, 1.0); // Randomize the tensor

    // input.print_tensor(); // Print the input tensor
    Tensor<float> mul_output = input * input2;   // THIS IS THE PROBLEM
    Tensor<float>* relu_output = relu(&mul_output); // Apply the relu function
    Tensor<float>* softmax_output = softmax(relu_output); // Apply the sofmax function
    // softmax_output.print_tensor(); // Print the output tensor
    backward(softmax_output); // Perform the backward pass

    cout << "Gradients with respect to the input tensor:" << endl;
    softmax_output->print_tensor(); // Print the gradients with respect to the input tensor
    relu_output->print_tensor(); // Print the gradients with respect to the input tensor
    mul_output.print_tensor(); // Print the gradients with respect to the input tensor
    // input2.print_tensor(); // Print the gradients with respect to the input tensor
    // input.print_tensor(); // Print the gradients with respect to the input tensor

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}