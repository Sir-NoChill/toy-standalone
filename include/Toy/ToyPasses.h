//===- ToyPasses.h - Toy passes  ------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef TOY_TOYPASSES_H
#define TOY_TOYPASSES_H

#include "Toy/ToyDialect.h"
#include "Toy/ToyOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace toy {
#define GEN_PASS_DECL
#include "Toy/ToyPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "Toy/ToyPasses.h.inc"
} // namespace toy
} // namespace mlir

#endif
