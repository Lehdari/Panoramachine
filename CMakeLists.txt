cmake_minimum_required(VERSION 3.14)

project(image_demorphing)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_subdirectory(ext)


# Add headers and sources
add_subdirectory(include)
add_subdirectory(src)


# Add image demorphing benchmark test target
add_executable(image_demorphing
    ${IMAGE_DEMORPHING_HEADERS}
    ${IMAGE_DEMORPHING_SOURCES}
)

target_link_libraries(image_demorphing
    PUBLIC
        img_demorph_bm
)

target_compile_definitions(image_demorphing
    PUBLIC
        IMAGE_DEMORPHING_TEST_RES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/res/"
)
