//===- ToyTypes.td - Toy dialect types -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TOY_TYPES
#define TOY_TYPES

include "mlir/IR/AttrTypeBase.td"
include "Toy/ToyDialect.td"

//===----------------------------------------------------------------------===//
// Toy type definitions
//===----------------------------------------------------------------------===//

class Toy_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Toy_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Toy_CustomType : Toy_Type<"Custom", "custom"> {
    let summary = "Toy custom type";
    let description = "Custom type in toy dialect";
    let parameters = (ins StringRefParameter<"the custom value">:$value);
    let assemblyFormat = "`<` $value `>`";
}

#endif // TOY_TYPES
