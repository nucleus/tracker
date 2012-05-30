PROJECT(boxfilt)

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

INCLUDE(FindCUDA)

INCLUDE_DIRECTORIES(${CUDA_SDK_ROOT_DIR}/../shared/inc)
INCLUDE_DIRECTORIES(${CUDA_SDK_ROOT_DIR}/common/inc)
INCLUDE_DIRECTORIES(./src)
LINK_DIRECTORIES(${CUDA_SDK_ROOT_DIR}/lib)
#LINK_LIBRARIES("-L${CUDA_SDK_ROOT_DIR}/lib -lcutil")

#MESSAGE("${CUDA_SDK_ROOT_DIR}")
#Include FindCUDA script

CUDA_ADD_EXECUTABLE(tracker ./src/main.cpp ./src/ForegroundSegmenter.cpp ./src/BallDetection.cpp ./src/VideoBackend.cpp ./src/ForegroundSegmenter.cu)
TARGET_LINK_LIBRARIES(tracker opencv_core)
TARGET_LINK_LIBRARIES(tracker opencv_imgproc)
TARGET_LINK_LIBRARIES(tracker opencv_highgui)
TARGET_LINK_LIBRARIES(tracker opencv_video)

set(CMAKE_CXX_FLAGS "-std=c++0x -O3")
set(CUDA_NVCC_FLAGS "-arch=sm_20")