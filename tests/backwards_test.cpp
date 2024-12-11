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
    Tensor<float> input({16, 4});

    input.randomize_tensor(-1.0, 1.0); // Randomize the tensor
    input.print_tensor(); // Print the input tensor
    // Tensor<float> relu_output = relu(&input); // Apply the relu function
    Tensor<float> softmax_output = softmax(&input); // Apply the relu function

    softmax_output.print_tensor(); // Print the output tensor

    // backward(&softmax_output); // Perform the backward pass

    // cout << "Gradients with respect to the input tensor:" << endl;
    // softmax_output.print_tensor(); // Print the gradients with respect to the input tensor

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}