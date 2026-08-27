cmake_minimum_required(VERSION 3.16)

project(video_tracking_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_library(video_processing
    src/video_reader.cpp
    src/lucas_kanade.cpp
    src/farneback.cpp
    src/postprocessing.cpp
)

target_link_libraries(
    video_processing
    PUBLIC
    ${OpenCV_LIBS}
)

add_executable(
    run_lucas_kanade
    apps/run_lucas_kanade.cpp
)

target_link_libraries(
    run_lucas_kanade
    PRIVATE
    video_processing
)

add_executable(
    run_farneback
    apps/run_farneback.cpp
)

target_link_libraries(
    run_farneback
    PRIVATE
    video_processing
)

add_executable(
    run_postprocessing
    apps/run_postprocessing.cpp
)

target_link_libraries(
    run_postprocessing
    PRIVATE
    video_processing
)

add_executable(
    check_opencv_cuda
    apps/check_opencv_cuda.cpp
)

target_link_libraries(
    check_opencv_cuda
    PRIVATE
    ${OpenCV_LIBS}
)