cmake_minimum_required(VERSION 3.8)
project(face_recognizer_eigenfaces)

find_package(OpenCV REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(SOURCE_FILES main.cpp eigenfaces.cpp helper.cpp)
add_executable(opencv ${SOURCE_FILES})

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(opencv ${OpenCV_LIBS})