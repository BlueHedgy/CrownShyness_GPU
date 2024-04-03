cmake_minimum_required(VERSION 3.21)

# This is your project statement. You should always list languages;
# Listing the version is nice here since it sets lots of useful variables
project(
  CrownShyness
  VERSION 1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_FLAGS "-g") 


add_subdirectory(LavaCake)
include_directories(headers)

set(SOURCES 
src/jitteredGrid.cpp
src/ultils.cpp
src/include_stb.cpp
)


# Adding something we can run - Output name matches target name
add_executable(CrownShyness src/main.cpp ${SOURCES})
target_link_libraries(CrownShyness LavaCake)

