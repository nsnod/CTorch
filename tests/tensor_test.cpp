#include <gtest/gtest.h>
#include "../src/tensor.h"

int test1() {
    std::cout << "test" << std::endl;
    return 1;
}

TEST(ExampleTest, OneIsOne) {
    int result = test1();
    EXPECT_EQ(result, 1) << "Expected result to be 1, but got " << result;
}

int main(int argc, char **argv) {

    //Tensor<float> test1;
    //test1.reshape({3,1});

    //test1.randomize_tensor(-1, 1);
    //test1.print_tensor();

    // Tensor<float> test2({1,10});

    // test2.print_tensor();

    // Tensor<float> test3 = test1 * test2;

    // test3.print_tensor();

    Tensor<float> inputTensor({2, 3}); // Batch size of 2, number of classes is 3, test made by CHATGPT HEE HEE
        inputTensor.data_->data_ = {
            1.0, 2.0, 3.0, // First sample
            1.0, 2.0, 3.0  // Second sample
    };

    // Print the input tensor
    std::cout << "Input Tensor:" << std::endl;
    inputTensor.print_tensor();

    // Step 2: Apply the softmax function
    Tensor<float> outputTensor = inputTensor.softmax(&inputTensor);

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
    inputTensor.softmax_backward(&inputTensor, &outputTensor);

    // Print the gradients with respect to the input tensor
    std::cout << "Input Tensor Gradients after Softmax Backward:" << std::endl;
    inputTensor.grad_->print();





    //cout << "This is the value at (1, 3) is " << test1[{2,0}] << endl;

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}