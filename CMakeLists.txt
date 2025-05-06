cmake_minimum_required(VERSION 3.10)
project(ImageFilters)

# Set default build type to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Set C++17 as the standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add executable
add_executable(filter 
    filter.cpp
    bmplib.cpp
)

# Include directories
target_include_directories(filter PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Set output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Link with required libraries
target_link_libraries(filter) 