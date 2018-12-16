cmake_minimum_required(VERSION 3.6)

enable_language(CXX)

set(ANACONDA_PATH $ENV{CONDA_PREFIX})

include_directories(${ANACONDA_PATH}/include)
include_directories(${PROJECT_SOURCE_DIR}/include)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/src/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/src/lib)

add_subdirectory(bin)
add_subdirectory(common)
add_subdirectory(dataset)
add_subdirectory(model)
add_subdirectory(ops)
