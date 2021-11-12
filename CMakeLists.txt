cmake_minimum_required(VERSION 3.14)

project(image_demorphing)

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


# Add image demorphing benchmark test target
add_executable(image_demorphing
    ${IMAGE_DEMORPHING_HEADERS}
    ${IMAGE_DEMORPHING_SOURCES}
)

target_include_directories(image_demorphing
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(image_demorphing
    PUBLIC
        img_demorph_bm
        OpenMP::OpenMP_CXX
        ${OpenCV_LIBS}
        Eigen3::Eigen
)

target_compile_definitions(image_demorphing
    PUBLIC
        IMAGE_DEMORPHING_RES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/res/"
)
