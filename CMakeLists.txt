cmake_minimum_required(VERSION 3.10)

project(CTorch)

set(CMAKE_CXX_STANDARD 11)

include_directories(src)

set(SOURCES
    main.cpp
<<<<<<< HEAD
    src/tensor.cpp
=======
    src/linear.cpp
>>>>>>> main
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

    add_executable(tensorTests tests/tensorTests.cpp)
    add_executable(arrayTests tests/arrayTests.cpp)
    add_executable(linear_test tests/linear_test.cpp)

    target_link_libraries(tensorTests gtest gtest_main)
    target_link_libraries(arrayTests gtest gtest_main)
    target_link_libraries(linear_test gtest gtest_main)


    add_test(NAME TensorTests COMMAND tensorTests)
    add_test(NAME ArrayTests COMMAND arrayTest)
    add_test(NAME LinearTest COMMAND linearTest)

endif()