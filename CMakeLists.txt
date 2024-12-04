cmake_minimum_required(VERSION 3.10)

project(CTorch)

set(CMAKE_CXX_STANDARD 11)

include_directories(src)

set(SOURCES
    main.cpp
    src/linear.cpp
)

set(HEADERS
    src/array.h
    src/linear.h
    src/tensor.h
)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_link_libraries(${PROJECT_NAME} ${MPI_LIBRARIES})

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

    target_link_libraries(tensorTests gtest gtest_main ${MPI_LIBRARIES})
    target_link_libraries(arrayTests gtest gtest_main ${MPI_LIBRARIES})

    add_test(NAME TensorTests COMMAND tensorTests)
    add_test(NAME ArrayTests COMMAND arrayTests)

endif()