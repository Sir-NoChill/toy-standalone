cmake_minimum_required(VERSION 3.20)
project(ToyBase)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

cmake_policy(SET CMP0057 NEW)

# Ensure tool dependencies.
include(ExternalProject) # Required to download and build external projects (i.e. ANTLR).
find_package(Git REQUIRED) # Need git to download ANTLR through ExternalProject.
find_package(Java COMPONENTS Runtime REQUIRED) # Need java to run ANTLR, but only the runtime.

# Ensure we have LLVM.
include("${CMAKE_SOURCE_DIR}/cmake/get_mlir.cmake")

## Build our dialect ##

# We store our dialect in lib and include
add_subdirectory("${CMAKE_SOURCE_DIR}/lib")
add_subdirectory("${CMAKE_SOURCE_DIR}/include")

# Add the dialect subdirectories
add_subdirectory(dialect)

## Setup ANTLR ##

# Link against the pthreads library to make std::call_once 
# in generated ANTLR code to run without producing system errors
# (see issue https://github.com/antlr/antlr4/issues/3708).
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

# Set C++ standards.
set(CMAKE_CXX_STANDARD 17)

# Add CXX flags.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

# Include cmake utilities.
include("${CMAKE_SOURCE_DIR}/cmake/symlink_to_bin.cmake")

# Grab ANTLR.
include("${CMAKE_SOURCE_DIR}/cmake/get_antlr.cmake")

# Set up paths and info to generate ANTLR sources with.
set(GRAMMAR_NAME "Toy")
set(ANTLR_NAMESPACE "toy")

# Generate sources.
include("${CMAKE_SOURCE_DIR}/cmake/antlr_generate.cmake")

## Build our compiler ##

# Include project headers.
include_directories("${CMAKE_SOURCE_DIR}/include")

# Add the source directory.
add_subdirectory("${CMAKE_SOURCE_DIR}/src")


# Add the runtime directory.
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/runtime")

