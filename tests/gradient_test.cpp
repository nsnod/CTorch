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
    Tensor<float> test({3, 4});
    Tensor<float> test2({3, 4});

    test.prev_->print();
    test += 1;
    test2 += 3;
    test += test2;
    cout << "Expect 4" << endl;
    test.print_tensor();

    test.prev_->print();
    test -= 2;
    cout << "Expect 2" << endl;
    test.print_tensor();

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}