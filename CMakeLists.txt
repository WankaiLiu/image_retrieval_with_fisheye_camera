cmake_minimum_required(VERSION 2.8.3)

project(image_retrieval)

#set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_BUILD_TYPE "Debug")
#set(CMAKE_CXX_FLAGS "-std=c++11 -O3 -fPIC")
set(CMAKE_CXX_FLAGS "-std=c++11 -fPIC")
find_package(OpenCV REQUIRED)
#find_package(Boost REQUIRED COMPONENTS filesystem program_options system)
#include_directories(${Boost_INCLUDE_DIRS})

aux_source_directory("src/" SRC_FILES)
aux_source_directory("src/DBoW/" DBOW_FILES)
aux_source_directory("src/DUtils/" DUTILS_FILES)
aux_source_directory("src/DVision/" DVISION_FILES)
#SET(UTILITY_PATH "src/utility/")
#aux_source_directory(${UTILITY_PATH} UTILITY_FILES)

#include_directories(${EIGEN3_INCLUDE_DIR})
include_directories("src/")
#add_library(bow  SRC_FILES DBOW_FILES)

set(SRC_CODE ${SRC_FILES} ${DBOW_FILES} ${DUTILS_FILES} ${DVISION_FILES} ${UTILITY_FILES})
add_library(image_retrieval SHARED ${SRC_CODE})
add_library(imdb_sdk_lib_shared SHARED ${SRC_CODE})
#target_link_libraries(image_retrieval)
#MESSAGE("Boost_LIBRARIES ${Boost_LIBRARIES} ${Boost_LIBS}")
add_executable(image_retrieval_test image_retrieval_test.cpp)
add_executable(imdb_sdk_test imdb_sdk_test.cpp)
target_link_libraries(image_retrieval_test ${OpenCV_LIBS} image_retrieval)
target_link_libraries(imdb_sdk_test  ${OpenCV_LIBS} imdb_sdk_lib_shared)


