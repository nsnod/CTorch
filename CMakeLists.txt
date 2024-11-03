cmake_minimum_required(VERSION 3.10)

project(CTorch)

set(CMAKE_CXX_STANDARD 11)

include_directories(src)

set(SOURCES
    main.cpp
    src/linear.cpp
    src/tensor.cpp
)

set(HEADERS
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

    target_link_libraries(tensorTests gtest gtest_main)

    add_test(NAME TensorTests COMMAND tensorTests)
endif()