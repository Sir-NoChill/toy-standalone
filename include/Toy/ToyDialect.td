//===- ToyDialect.td - Toy dialect -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TOY_DIALECT
#define TOY_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Toy dialect definition.
//===----------------------------------------------------------------------===//

def Toy_Dialect : Dialect {
    let name = "toy";
    let summary = "A toy out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.

	This particular dialect also works with the University of Alberta
	CMPUT 415 teaching materials in this repository, including antlr
	integration and mlir-based code generation for a complete language.

	For teaching purposes, we implement the Toy language from the mlir
	[toy tutorial](https://mlir.llvm.org/docs/Tutorials/Toy)
    }];
    let cppNamespace = "::mlir::toy";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

//===----------------------------------------------------------------------===//
// Base standalone operation definition.
//===----------------------------------------------------------------------===//

class Toy_Op<string mnemonic, list<Trait> traits = []> :
        Op<Toy_Dialect, mnemonic, traits>;

#endif // TOY_DIALECT
