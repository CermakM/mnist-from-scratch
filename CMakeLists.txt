cmake_minimum_required(VERSION 3.8)
project(mnist_from_scratch)

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES src/main.cpp src/dataset.cpp src/dataset.h src/common.h tests/main.cpp)
add_executable(mnist_from_scratch ${SOURCE_FILES})
