cmake_minimum_required(VERSION 3.6)

# Boost

FIND_PACKAGE( Boost REQUIRED COMPONENTS system filesystem )

set (Boost_NO_SYSTEM_PATHS ON)
set (Boost_USE_MULTITHREADED ON)
set (Boost_USE_STATIC_LIBS ON)
set (Boost_USE_STATIC_RUNTIME OFF)
set (BOOST_ALL_DYN_LINK OFF)


if (Boost_FOUND)

    include_directories(
        SYSTEM ${Boost_INCLUDE_DIR}
    )

endif()


link_directories(SYSTEM "/usr/lib64/")  # libcblas
link_directories(SYSTEM "${ANACONDA_PATH}/lib")  # libcblas

find_library(CBLAS NAMES cblas libcblas)  # libcblas

set(EXTRA_LIBS ${EXTRA_LIBS} dataset model cblas)


# target

include_directories("${PROJECT_SOURCE_DIR}/src")

add_executable(mnist main.cpp)
add_executable(mnist-evaluate evaluate.cpp)

target_link_libraries(mnist ${EXTRA_LIBS} ${Boost_LIBRARIES})
target_link_libraries(mnist-evaluate ${EXTRA_LIBS} ${Boost_LIBRARIES})


# make links in the source directory to the built executables
add_custom_command(TARGET mnist POST_BUILD
    COMMAND ln -sfn build/src/bin/mnist mnist
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMENT "Creating link to executable `build/src/bin/mnist` ..."
)

add_custom_command(TARGET mnist-evaluate POST_BUILD
    COMMAND ln -sfn build/src/bin/mnist-evaluate mnist-evaluate
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMENT "Creating link to executable `build/src/bin/mnist-evaluate` ..."
)
