#include <gtest/gtest.h>
#include "../src/array.h"

using namespace std;

// TEST(ExampleTest, OneIsOne) {
//     int result = 1;
//     EXPECT_EQ(result, 1) << "Expected result to be 1, but got " << result;
// }

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    Array<float> myArray({3, 3});

    cout << myArray.getDim() << " " << myArray.getSize() << endl;

    myArray.randomize(-1.0, 2.0);

    myArray.print();

    return RUN_ALL_TESTS();
}