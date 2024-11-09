#include <gtest/gtest.h>
#include "../src/array.h"

using namespace std;

// TEST(ExampleTest, OneIsOne) {
//     int result = 1;
//     EXPECT_EQ(result, 1) << "Expected result to be 1, but got " << result;
// }

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    Array<float> myArray({10, 1});

    cout << myArray.getDim() << " " << myArray.getSize() << endl;


    myArray.randomize(-1.0, 1.0);

    vector<float> testData = myArray.getData();

    for (int i = 0; i < testData.size(); i++) {
        cout << testData[i] << " ";
    }
    cout << endl << endl;
    
    myArray.print();



    // cout << endl << endl << endl;
    // //testing flat index
    // vector<int> shape = {3, 3};
    // Array<float> array(shape);

    // vector<int> strides = array.getStrides();
    // cout << "Strides: ";
    // for (int s : strides) cout << s << " ";
    // cout << endl;

    // vector<int> testIndices1 = {0, 0};
    // vector<int> testIndices2 = {0, 1};
    // vector<int> testIndices3 = {1, 2};

    // cout << "Flat index for {0, 0}: " << array.flatIndex(testIndices1) << endl;
    // cout << "Flat index for {0, 1}: " << array.flatIndex(testIndices2) << endl;
    // cout << "Flat index for {1, 2}: " << array.flatIndex(testIndices3) << endl;


    // cout << endl << endl << endl;
    // //testing the at
    // vector<int> shape2 = {3, 2};
    // Array<float> array2(shape);
    // array2.randomize(-1.0, 1.0);
    // array2.print();
    // cout << "print above" << endl;
    // cout << "direct .at accessing below" << endl;
    // cout << array2.at({0, 0}) << endl;
    // cout << array2.at({1, 1}) << endl;


    return RUN_ALL_TESTS();
}
