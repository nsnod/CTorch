#include <gtest/gtest.h>
#include "../src/array.h"

TEST(ExampleTest, OneIsOne) {
    int result = 1;
    EXPECT_EQ(result, 1) << "Expected result to be 1, but got " << result;
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}