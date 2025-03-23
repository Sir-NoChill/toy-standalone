add_mlir_dialect(ToyOps toy)
add_mlir_doc(ToyDialect ToyDialect Toy/ -gen-dialect-doc)
add_mlir_doc(ToyOps ToyOps Toy/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS ToyPasses.td)
mlir_tablegen(ToyPasses.h.inc --gen-pass-decls)
add_public_tablegen_target(MLIRToyPassesIncGen)
