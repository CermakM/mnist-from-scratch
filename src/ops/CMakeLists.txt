cmake_minimum_required(VERSION 3.6)

include_directories(
        "${PROJECT_SOURCE_DIR}/src"
)

add_library(ops SHARED ops.cpp)
