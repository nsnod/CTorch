cmake_minimum_required(VERSION 3.10)

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
    src/mnist/mnist.cpp
)

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

# Enable testing if required
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

    # Add test executables
    add_executable(tensorTests tests/tensorTests.cpp)
    add_executable(arrayTests tests/arrayTests.cpp)
    add_executable(linear_test tests/linear_test.cpp)

    # Link Google Test libraries to test executables
    target_link_libraries(tensorTests gtest gtest_main)
    target_link_libraries(arrayTests gtest gtest_main)
    target_link_libraries(linear_test gtest gtest_main)

    # Add test cases
    add_test(NAME TensorTests COMMAND tensorTests)
    add_test(NAME ArrayTests COMMAND arrayTests)
    add_test(NAME LinearTest COMMAND linear_test)
endif()
