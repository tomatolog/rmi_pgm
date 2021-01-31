cmake_minimum_required(VERSION 3.10)
project(rmi_pgm)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

add_subdirectory(PGM-index)
add_subdirectory(RadixSpline)

include_directories(
        ${CMAKE_SOURCE_DIR}
        cpp-btree/btree
)

add_executable(benchmark main.cpp rmis/fb.cpp rmis/wiki.cpp rmis/osm.cpp rmis/books.cpp)
