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