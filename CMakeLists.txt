cmake_minimum_required(VERSION 2.6)

project(seg_net)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

set(CUDA_NVCC_FLAGS $(CUDA_NVCC_FLAGS);-std=c++11;-g;-G;-gencode;arch=compute_30;code=sm_30)

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
#include_directories(/media/agx/sd/software/TensorRT-7.2.2.3/targets/x86_64-linux-gnu/include/)
#link_directories(/media/agx/sd/software/TensorRT-7.2.2.3/targets/x86_64-linux-gnu/lib/)
include_directories(/usr/include/aarch64-linux-gnu)
link_directories(/usr/lib/aarch64-linux-gnu)

set(CMAKE_CXX_FLAGS "$(CMAKE_CXX_FLAGS) -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

add_executable(seg_net ${PROJECT_SOURCE_DIR}/seg_net.cpp)
target_link_libraries(seg_net nvinfer)
target_link_libraries(seg_net cudart)

add_definitions(-O2 -pthread)

#opencv library
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(seg_net ${OpenCV_LIBS})
