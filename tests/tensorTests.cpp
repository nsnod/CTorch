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

    Tensor<float> test({3,2});

    test.randomize_tensor(-1, 1);

    test.print_tensor();

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}