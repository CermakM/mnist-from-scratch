cmake_minimum_required(VERSION 3.1)

enable_language(CXX)

set(ANACONDA_PATH $ENV{CONDA_PREFIX})

include_directories(${ANACONDA_PATH}/include)
include_directories(${PROJECT_SOURCE_DIR}/include)

add_subdirectory(bin)
add_subdirectory(common)
add_subdirectory(dataset)
add_subdirectory(model)
add_subdirectory(ops)
