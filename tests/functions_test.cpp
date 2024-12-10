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
    /*
    // test softmax
    Tensor<float> inputTensor({2, 3}); // Batch size of 2, number of classes is 3, test made by CHATGPT HEE HEE
        inputTensor.data_->data_ = {
            1.0, 2.0, 3.0, // First sample
            1.0, 2.0, 3.0  // Second sample
    };

    // Print the input tensor
    std::cout << "Input Tensor:" << std::endl;
    inputTensor.print_tensor();

    // Step 2: Apply the softmax function
    Tensor<float> outputTensor = softmax(&inputTensor);

    // Print the output tensor after softmax
    std::cout << "Output Tensor after Softmax:" << std::endl;
    outputTensor.print_tensor();

    // Step 3: Assume some gradient coming from the next layer
    outputTensor.grad_ = new Array<float>(outputTensor.shape_);
    outputTensor.grad_->data_ = {
        0.1f, 0.2f, 0.7f, // Gradient for first sample
        0.3f, 0.4f, 0.3f  // Gradient for second sample
    };

    // Print the gradients from the next layer
    std::cout << "Gradient from Next Layer:" << std::endl;
    outputTensor.grad_->print();

    // Step 4: Perform backward pass
    softmax_backward(&inputTensor, &outputTensor);

    // Print the gradients with respect to the input tensor
    std::cout << "Input Tensor Gradients after Softmax Backward:" << std::endl;
    inputTensor.grad_->print();


    // test relu
    Tensor<float> inputTensor2({2, 3}); // Batch size of 2, number of classes is 3, test made by CHATGPT HEE HEE
        inputTensor2.data_->data_ = {
            1.0, 2.0, 3.0, // First sample
            -1.0, 2.0, -3.0  // Second sample
    };

    // Print the input tensor
    std::cout << "Input Tensor:" << std::endl;

    // Step 2: Apply the relu function
    Tensor<float> outputTensor2 = relu(&inputTensor2);

    // Print the output tensor after relu
    std::cout << "Output Tensor after ReLU:" << std::endl;
    outputTensor2.print_tensor();
    */

    // test mean
    Tensor<float> inputTensor3({2, 3}); // Batch size of 2, number of classes is 3, test made by CHATGPT HEE HEE
        inputTensor3.data_->data_ = {
            1.0, 2.0, 3.0, // First sample
            -1.0, 2.0, -3.0  // Second sample
    };
    Tensor<float>* outputTensor3 = nullptr;
    outputTensor3 = mean(&inputTensor3);
    outputTensor3->print_tensor();

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}