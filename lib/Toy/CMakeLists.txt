add_mlir_dialect_library(MLIRToy
  Ops.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Toy

  DEPENDS
  ToyOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRFuncDialect
  MLIRAnalysis
  MLIRFunctionInterfaces
  MLIRParser
  MLIRSideEffectInterfaces
  MLIRTransforms
)
