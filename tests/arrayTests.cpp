#include <gtest/gtest.h>
#include "../src/array.h"

using namespace std;

// TEST(ExampleTest, OneIsOne) {
//     int result = 1;
//     EXPECT_EQ(result, 1) << "Expected result to be 1, but got " << result;
// }

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    Array<float> myArray({1, 5});

    cout << myArray.getDim() << " " << myArray.getSize() << endl;


    myArray.randomize(-1.0, 2.0);

    vector<float> testData = myArray.getData();

    for (int i = 0; i < testData.size(); i++) {
        cout << testData[i] << " ";
    }
    
    myArray.print();



    return RUN_ALL_TESTS();
}