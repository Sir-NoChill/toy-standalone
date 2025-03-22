# Gather our source files in this directory.
set(
  toy_rt_files
  "${CMAKE_CURRENT_SOURCE_DIR}/toy.c"
)

# Build our executable from the source files.
add_library(toyrt SHARED ${toy_rt_files})
target_include_directories(toyrt PUBLIC ${RUNTIME_INCLUDE})

# Symbolic link our library to the base directory so we don't have to go searching for it.
symlink_to_bin("toyrt")
