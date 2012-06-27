PROJECT(tracker)

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

INCLUDE(FindCUDA)

set(CUDA_NVCC_FLAGS "-arch=sm_20")

INCLUDE_DIRECTORIES(${CUDA_SDK_ROOT_DIR}/../shared/inc)
INCLUDE_DIRECTORIES(${CUDA_SDK_ROOT_DIR}/common/inc)
INCLUDE_DIRECTORIES(./src)
LINK_DIRECTORIES(${CUDA_SDK_ROOT_DIR}/lib)

FIND_PACKAGE(Boost)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})

CUDA_ADD_EXECUTABLE(tracker ./src/main.cpp ./src/ForegroundSegmenter.cpp ./src/BallDetection.cpp ./src/VideoBackend.cpp ./src/ForegroundSegmenter.cu ./src/threading.cpp)
TARGET_LINK_LIBRARIES(tracker opencv_core)
TARGET_LINK_LIBRARIES(tracker opencv_imgproc)
TARGET_LINK_LIBRARIES(tracker opencv_highgui)
TARGET_LINK_LIBRARIES(tracker opencv_video)
TARGET_LINK_LIBRARIES(tracker ${Boost_LIBRARIES} )
TARGET_LINK_LIBRARIES(tracker boost_thread-mt)
TARGET_LINK_LIBRARIES(tracker boost_program_options)

set(CMAKE_CXX_FLAGS "-std=c++0x -O3")
