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

    Tensor<float> test({3});
    Tensor<float> mulMe({3});

    vector<int> shapeTest = test.getData()->getShape();

    cout << "SHAPE" << endl;
    for (int i = 0; i < shapeTest.size(); i++) {
        cout << shapeTest[i] << " ";
    }
    cout << endl;

    test.randomize_tensor(-1, 1);

    test.print_tensor();

    //seg fault
    // test.tensorMulData(test.getData(), mulMe.getData());

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}