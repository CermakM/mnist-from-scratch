cmake_minimum_required(VERSION 3.1)

set(PROJECT_NAME mnist_from_scratch)

project(${PROJECT_NAME})


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/src/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/src/lib)

add_subdirectory(src)
#add_subdirectory(tests)
