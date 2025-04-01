//===- ToyDialect.cpp - Toy dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Toy/Dialect.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/Interfaces/FunctionImplementation.h"

using namespace mlir;
using namespace mlir::toy;

#include "Toy/Dialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Toy dialect
//===----------------------------------------------------------------------===//

void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Toy/Ops.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Toy ConstantOp
//===----------------------------------------------------------------------===//

// This is our custom builder defined in the tblgen file. You can see where in the
//  ConstantOp builders section. This is the builder defined for a single value of 
//  type double
void ConstantOp::build(mlir::OpBuilder &odsBuilder, mlir::OperationState &odsState, 
    double value) {
  auto dataType = RankedTensorType::get({}, odsBuilder.getF64Type());  
  auto dataAttribute = DenseElementsAttr::get(dataType, value);

  ConstantOp::build(odsBuilder, odsState, dataType, dataAttribute);
}

// This is our custom printer and parser for the constant op.
// We can't use the default builder/the declarative format
// because the value is defined in the attributes of the function rather than
// as a functional input of the op.
void ConstantOp::print(mlir::OpAsmPrinter &printer) {
  printer << " ";
  printer.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/ {"value"});
  printer << getValue();
}

// The parser corresponds to the ConstantOp::print function to parse the
// operation
mlir::ParseResult ConstantOp::parse(mlir::OpAsmParser &parser,
		       mlir::OperationState &result) {
  mlir::DenseElementsAttr value;
  // These parse operations are defined in such a way to be the inversion
  // of the boolean values so that we can easily chain them together
  // and exit on _failure_ rather than success.
  if (parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseAttribute(value, "value", result.attributes))
    return failure();

  result.addTypes(value.getType());
  return success();
}

// We need to verify that all constants end up being of rank > 1, so that we can
// ensure that we are within the language constraints. Easy enough to do with
// a little custom verifier
llvm::LogicalResult ConstantOp::verify() {
  // First ensure that if the return type is not an unranked tensor that the
  // attribute data shape matches the shape of the return type.
  // If we have an unranked tensor return type then we will be reshaping
  // the tensor according to its operation definition.
  auto resultType = llvm::dyn_cast<mlir::RankedTensorType>(getResult().getType());
  if (!resultType) return llvm::success();

  // Now that we know that we have a ranked tensor, let's check if the shapes are
  // the same
  auto attrType = llvm::cast<mlir::RankedTensorType>(getResult().getType());
  if (attrType.getRank() != resultType.getRank()) {
    return emitOpError("return type must match that of the attached value"
        "attribute: ")
      << attrType.getRank() << " != " << resultType.getRank();
  }

  // Now that we know the shapes are the same, we can check each dimension and
  // make sure that they are all the same as well.
  for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim) {
      if (attrType.getShape()[dim] != resultType.getShape()[dim]) {
	return emitOpError(
		  "Return type shape mismatch with the attrbute at dimension")
	        << dim << ": " << attrType.getShape()[dim] << " != "
	        << resultType.getShape()[dim];
	      
	}
    }

  // If we are still good, then we can continue
  return llvm::success();
}

//===----------------------------------------------------------------------===//
// Toy AddOp
//===----------------------------------------------------------------------===//

// Our custom builder for the AddOp
void AddOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, mlir::Value lhs, mlir::Value rhs) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands({lhs, rhs});
}

//===----------------------------------------------------------------------===//
// Toy FuncOp
//===----------------------------------------------------------------------===//

// Custom printer and parser for our function type
void FuncOp::print(mlir::OpAsmPrinter &printer) {
  // We can dispatch to the function op interface which provides a printer
  mlir::function_interface_impl::printFunctionOp(printer, *this, /*isVariadic=*/false, 
      getFunctionTypeAttrName(), getArgAttrsAttrName(), getResAttrsAttrName());
}

mlir::ParseResult FuncOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  auto buildFuncType = 
    [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes, 
       llvm::ArrayRef<mlir::Type> resultTypes, 
       mlir::function_interface_impl::VariadicFlag, std::string &) 
    { return builder.getFunctionType(argTypes, resultTypes); };

  return mlir::function_interface_impl::parseFunctionOp(parser, result, /*allowVariadic=*/false, 
      getFunctionTypeAttrName(result.name), buildFuncType, 
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void FuncOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, 
    StringRef name, FunctionType type, ArrayRef<NamedAttribute> attrs) {
  // The FunctionOpInterface provides a nice little build method that we can use
  buildWithEntryBlock(odsBuilder, odsState, name, type, attrs, type.getInputs());
}

#define GET_OP_CLASSES
#include "Toy/Ops.cpp.inc"
