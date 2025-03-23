# Gather our source files in this directory.
set(BACKEND "${CMAKE_CURRENT_SOURCE_DIR}/Backend")
set(AST "${CMAKE_CURRENT_SOURCE_DIR}/AST")
set(AST_PASSES "${CMAKE_CURRENT_SOURCE_DIR}/ASTPasses")
set(TOY "${CMAKE_CURRENT_SOURCE_DIR}/Toy")
set(
  TOY_BACKEND_FILES
    "${BACKEND}/BackEnd.cpp"
)

set(
  TOY_AST_FILES
  "${AST}/ASTBuilder.cpp"
)

set(
  toy_src_files
    "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
    "${TOY_BACKEND_FILES}"
    "${TOY_AST_FILES}"
)



# Build our executable from the source files.
add_executable(toyc ${toy_src_files})
target_include_directories(toyc PUBLIC ${ANTLR_GEN_DIR})

# Ensure that the antlr4-runtime is available.
add_dependencies(toyc antlr)

# Find the libraries that correspond to the LLVM components
# that we wish to use
set(LLVM_LINK_COMPONENTS Core Support)
llvm_map_components_to_libnames(llvm_libs core)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

# Add the MLIR, LLVM, antlr runtime and parser as libraries to link.
target_link_libraries(toyc PRIVATE
    parser
    antlr4-runtime
    ${llvm_libs}
    ${dialect_libs}
    )

# Symbolic link our executable to the base directory so we don't have to go searching for it.
symlink_to_bin("toyc")
