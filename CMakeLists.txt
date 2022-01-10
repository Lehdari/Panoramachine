cmake_minimum_required(VERSION 3.14)

project(panoramachine)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


# Find libraries
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

add_subdirectory(ext)


# Add headers and sources
add_subdirectory(include)
add_subdirectory(src)


# Add panoramachine executable target
add_executable(panoramachine
    ${PANORAMACHINE_HEADERS}
    ${PANORAMACHINE_SOURCES}
)

target_include_directories(panoramachine
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(panoramachine
    PUBLIC
        OpenMP::OpenMP_CXX
        ${OpenCV_LIBS}
        Eigen3::Eigen
        delaunay
)

target_compile_definitions(panoramachine
    PUBLIC
        PANORAMACHINE_RES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/res/"
)
