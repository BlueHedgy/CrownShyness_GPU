cmake_minimum_required(VERSION 3.21)

# This is your project statement. You should always list languages;
# Listing the version is nice here since it sets lots of useful variables
project(
  CrownShyness
  VERSION 1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Debug")

if (CMAKE_<LANG>_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "-Wall -Wno-sign-compare")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
add_compile_definitions("CMAKE_SOURCE_DIR=${CMAKE_SOURCE_DIR}")

add_subdirectory(LavaCake)
include_directories(headers)
include_directories(nlohmann)

set(SOURCES
src/forest_controls.cpp 
src/jitteredGrid.cpp
src/utils.cpp
src/include_stb.cpp
)


# Adding something we can run - Output name matches target name
add_executable(CrownShyness src/main.cpp ${SOURCES})
target_link_libraries(CrownShyness LavaCake)

