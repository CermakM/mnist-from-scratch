cmake_minimum_required(VERSION 3.1)

include_directories(
        "${PROJECT_SOURCE_DIR}/src"
)

add_library(model SHARED model.cpp)

target_link_libraries(model ops)
