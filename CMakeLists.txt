cmake_minimum_required(VERSION 3.16)

# Project name
project(CTorch)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)

# Set OpenCV directory to the path where OpenCVConfig.cmake is located
set(OpenCV_DIR "C:/opencv/build/install") # Adjust this path

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include directories for source files
include_directories(src)

# Set the source and header files
set(SOURCES
    # main.cpp
    main.cpp
    src/mnist/mnist.cpp
    src/backwards.cpp
    src/functions.cpp
)

# Header files
set(HEADERS
    src/array.h
    src/linear.h
    src/tensor.h
)

# Add the main executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Include OpenCV headers
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})

# Link against OpenCV libraries
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})

# Locate MPI package (C and CXX components)
find_package(MPI REQUIRED)

# Add include directories for MPI
include_directories(${MPI_INCLUDE_PATH})

# Link directories (optional)
link_directories("/opt/homebrew/lib")

# Add executable for main project
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Link MPI libraries
target_link_libraries(${PROJECT_NAME} PRIVATE MPI::MPI_C MPI::MPI_CXX)

# Enable testing if the option is set to ON
option(BUILD_TESTING "Enable testing" OFF)

if(BUILD_TESTING)
    enable_testing()

    # Add Google Test framework
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/release-1.12.1.tar.gz
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(googletest)


    # Add the test executables
    add_executable(tensor_test tests/tensor_test.cpp)
    add_executable(array_test tests/array_test.cpp)
    add_executable(linear_test tests/linear_test.cpp)
    add_executable(gradient_test tests/gradient_test.cpp)
    add_executable(backwards_test tests/backwards_test.cpp)
    add_executable(functions_test tests/functions_test.cpp) 

    # Link the test executables with GTest and MPI
    target_link_libraries(tensor_test PRIVATE gtest gtest_main MPI::MPI_C MPI::MPI_CXX)
    target_link_libraries(array_test PRIVATE gtest gtest_main MPI::MPI_C MPI::MPI_CXX)
    target_link_libraries(linear_test PRIVATE gtest gtest_main MPI::MPI_C MPI::MPI_CXX)
    target_link_libraries(gradient_test PRIVATE gtest gtest_main MPI::MPI_C MPI::MPI_CXX)
    target_link_libraries(backwards_test PRIVATE gtest gtest_main MPI::MPI_C MPI::MPI_CXX)
    target_link_libraries(functions_test PRIVATE gtest gtest_main MPI::MPI_C MPI::MPI_CXX) 

    # Add tests to CTest (allows ctest to run them)
    add_test(NAME tensor_test COMMAND tensor_test)
    add_test(NAME array_test COMMAND array_test)
    add_test(NAME linear_test COMMAND linear_test)
    add_test(NAME gradient_test COMMAND gradient_test)
    add_test(NAME backwards_test COMMAND backwards_test)
    add_test(NAME functions_test COMMAND functions_test)
endif()
