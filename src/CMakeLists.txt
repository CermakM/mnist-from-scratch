cmake_minimum_required(VERSION 3.8)

enable_language(CXX)


set(ANACONDA_PATH /home/macermak/anaconda3)


include_directories(${ANACONDA_PATH}/include)

add_subdirectory(bin)
add_subdirectory(common)
add_subdirectory(dataset)
