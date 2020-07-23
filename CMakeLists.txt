cmake_minimum_required(VERSION 2.8.3)

project(image_retrieval)

#set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_BUILD_TYPE "Debug")
#set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -fPIC")
set(CMAKE_CXX_FLAGS "-std=c++11 -fPIC")
set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/opencv/share/OpenCV/)
message("Current directory: " ${OpenCV_DIR})
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem program_options system)
message("OpenCV directory: " ${OpenCV_DIR})
include_directories(${Boost_INCLUDE_DIRS})

aux_source_directory("src/" SRC_FILES)
aux_source_directory("src/DBoW/" DBOW_FILES)
aux_source_directory("src/DUtils/" DUTILS_FILES)
aux_source_directory("src/DVision/" DVISION_FILES)

include_directories("src/")

set(SRC_CODE ${SRC_FILES} ${DBOW_FILES} ${DUTILS_FILES} ${DVISION_FILES} ${UTILITY_FILES})
add_library(imdb_sdk_lib SHARED ${SRC_CODE})
target_link_libraries(imdb_sdk_lib ${OpenCV_LIBS})
add_executable(imdb_sdk_test imdb_sdk_test.cpp)
target_link_libraries(imdb_sdk_test imdb_sdk_lib)


