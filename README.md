# CTorch
:3

# Description

# Documentation

## array.h
The array.h class is our custom class for dealing with N dimensional arrays.\
The array constructor takes a shape and creates an array with that many zeros in a one dimensional array.\
Even though it is stored in a one dimensional array we access it like its some N dimentional array with our special indexing - strides. Users do not interact with the Array class - it is a custom "datatype" to be used by tensor.


# Building
Turn the testing option in CMakeLists.txt OFF or ON if you want to run the tests. Make the script executable and run.
> chmod +x build.sh
> ./build.sh


run output files via
>./build/OUTPUTFILENAME