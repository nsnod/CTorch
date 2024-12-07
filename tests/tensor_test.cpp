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
    Tensor<float> test1;
    test1.reshape({3,1});

    test1.randomize_tensor(-1, 1);
    test1.print_tensor();

    Tensor<float> test2({1,10});

    test2.print_tensor();

    Tensor<float> test3 = test1 * test2;

    test3.print_tensor();

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}