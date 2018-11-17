cmake_minimum_required(VERSION 3.8)

FIND_PACKAGE( Boost REQUIRED COMPONENTS system filesystem )

set (Boost_NO_SYSTEM_PATHS ON)
set (Boost_USE_MULTITHREADED ON)
set (Boost_USE_STATIC_LIBS ON)
set (Boost_USE_STATIC_RUNTIME OFF)
set (BOOST_ALL_DYN_LINK OFF)

set(EXTRA_LIBS ${EXTRA_LIBS} dataset)


if (Boost_FOUND)

    include_directories(
        SYSTEM ${Boost_INCLUDE_DIR}
    )

endif()


include_directories("${PROJECT_SOURCE_DIR}/src")


add_executable(mnist main.cpp)

target_link_libraries(mnist ${EXTRA_LIBS} ${Boost_LIBRARIES})