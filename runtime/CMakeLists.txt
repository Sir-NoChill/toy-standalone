cmake_minimum_required(VERSION 3.20)
project(ToyRuntime)

# Set a variable for the runtime include directory.
set(RUNTIME_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Build the actual runtime.
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/src")

