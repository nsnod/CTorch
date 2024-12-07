cmake_minimum_required(VERSION 3.10)

project(CTorch)

set(CMAKE_CXX_STANDARD 11)

include_directories(src)

set(SOURCES
    main.cpp
    src/backwards.cpp
    src/functions.cpp
)

set(HEADERS
    src/array.h
    src/linear.h
    src/tensor.h
)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

option(BUILD_TESTING "Enable testing" OFF)

if(BUILD_TESTING)
    enable_testing()

    include(FetchContent)
    FetchContent_Declare(
      googletest
      URL https://github.com/google/googletest/archive/release-1.12.1.tar.gz
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(googletest)

    add_executable(tensor_test tests/tensor_test.cpp)
    add_executable(array_test tests/array_test.cpp)
    add_executable(linear_test tests/linear_test.cpp)
    add_executable(gradient_test tests/gradient_test.cpp)
    add_executable(backwards_test tests/backwards_test.cpp)
    add_executable(functions_test tests/functions_test.cpp) # Add the new functions test

    target_link_libraries(tensor_test gtest gtest_main)
    target_link_libraries(array_test gtest gtest_main)
    target_link_libraries(linear_test gtest gtest_main)
    target_link_libraries(gradient_test gtest gtest_main)
    target_link_libraries(backwards_test gtest gtest_main)
    target_link_libraries(functions_test gtest gtest_main) # Link the new test with gtest

    add_test(NAME tensor_test COMMAND tensor_test)
    add_test(NAME array_test COMMAND array_test)
    add_test(NAME linear_test COMMAND linear_test)
    add_test(NAME gradient_test COMMAND gradient_test)
    add_test(NAME backwards_test COMMAND backwards_test)
    add_test(NAME functions_test COMMAND functions_test) # Add the new test to CTest
endif()