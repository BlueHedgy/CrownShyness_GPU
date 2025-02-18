cmake_minimum_required(VERSION 3.21)

# This is your project statement. You should always list languages;
# Listing the version is nice here since it sets lots of useful variables
project(
  CrownShyness
  VERSION 1.0 LANGUAGES CXX)
  
set(CMAKE_BUILD_TYPE "Debug")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (CMAKE_<LANG>_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "-Wall -Wno-sign-compare")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
add_compile_definitions("CMAKE_SOURCE_DIR=${CMAKE_SOURCE_DIR}")

add_subdirectory(LavaCake)
include_directories(nlohmann)


# FLIP THIS FLAG TO USE CPU VERSION
set(USE_CUDA TRUE)

include_directories(headers_GPU)
include_directories(headers)

# Adding something we can run - Output name matches target name
if (USE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  include_directories(${CUDA_INCLUDE_DIRS})
  set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")  # enable cuda-gdb
  
  # include_directories(headers_GPU)
  set(SOURCES_GPU
  src_GPU/jitteredGrid.cu
  src_GPU/utils.cu
  src_GPU/forest_controls.cu
  src_GPU/include_stb.cu
  )
  add_executable(CrownShyness src_GPU/main.cu ${SOURCES_GPU})

else()
  # include_directories(headers)
  set(SOURCES_CPU
    src/jitteredGrid.cpp
    src/forest_controls.cpp 
    src/utils.cpp
    src/include_stb.cpp
  )
  add_executable(CrownShyness src/main.cpp ${SOURCES} ${SOURCES_CPU})

endif()

set_target_properties(CrownShyness PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(CrownShyness LavaCake)

