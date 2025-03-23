//===- ToyDialect.cpp - Toy dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Toy/ToyDialect.h"
#include "Toy/ToyOps.h"
#include "Toy/ToyTypes.h"

using namespace mlir;
using namespace mlir::toy;

#include "Toy/ToyOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Toy dialect.
//===----------------------------------------------------------------------===//

void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Toy/ToyOps.cpp.inc"
      >();
  registerTypes();
}
