cmake_minimum_required(VERSION 3.8)

enable_language(CXX)


set(ANACONDA_PATH $ENV{CONDA_PREFIX})


include_directories(${ANACONDA_PATH}/include)

add_subdirectory(bin)
add_subdirectory(common)
add_subdirectory(dataset)
add_subdirectory(model)
add_subdirectory(ops)
