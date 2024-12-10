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
    Tensor<float> test({16, 4});
    Tensor<float> test2({16, 4});
    Array<float> data({16, 4});
    Array<float>* output;

    data 
    output = data * 2;
    output->print();

    test.print_tensor();
    // test += 1;
    // test2 += 3;
    // test.clear_prev();
    // test += test2;
    // cout << "Expect 4" << endl;
    // test.print_tensor();

    // (*test.prev_)[1]->print();    
    // test -= 2;
    // cout << "Expect 2" << endl;
    // test.print_tensor();

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}