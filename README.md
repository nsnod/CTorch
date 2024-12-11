# CTorch
<a name="top"></a>
<div align="center">
<img src = "./src/images/CTorch_Library.png">

# Description
Understanding PyTorch, a potent machine learning package, and investigating methods to speed up its calculations through parallelization are the main goals of this project. In order to maximize PyTorch's usefulness, we aim to locate performance bottlenecks in its operations and apply effective parallel processing strategies.

Our work is based on CTorch, an open-source project that focuses on GPU acceleration by re-implementing parts of PyTorch in C++. We hope to significantly improve the speed and effectiveness of PyTorch's matrix multiplication, backpropagation, tensors, linear functions, etc. by utilizing the ideas and design of CTorch.

# Documentation

## array.h
The array.h class is our custom class for dealing with N dimensional arrays.\
The array constructor takes a shape and creates an array with that many zeros in a one dimensional array.\
Even though it is stored in a one dimensional array we access it like its some N dimentional array with our special indexing - strides. Users do not interact with the Array class - it is a custom "datatype" to be used by tensor.


# Building
Turn the testing option in CMakeLists.txt OFF or ON if you want to run the tests. Make the script executable and run.
> chmod +x build.sh
> ./build.sh


Run output files via
>./build/OUTPUTFILENAME

Our tensors functions are inherently parallel so when running parallel code use the mpi run function.
>mpirun -np (# of processes) ./build/OUTPUTFILENAME
