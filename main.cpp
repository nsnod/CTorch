// driver code for testing
#include "src/linear.h"
#include "src/tensor.h"
#include "src/array.h"
#include <iostream>
int main () {
    std::vector<int> shape = {2, 2};  // Shape for a 2x2 tensor
    Tensor<float> tensor(shape);      // Instantiate Tensor object

    tensor.randomize_tensor(0.0f, 1.0f);
    
    tensor.resetZeroData();
    tensor.resetZeroGrad();  // Randomize contents between 0 and 1
    tensor.print_tensor();
    
    tensor.print_tensor();                // Print tensor contents

    return 0;

}