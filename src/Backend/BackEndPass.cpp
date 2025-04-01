#include "Backend/BackEnd.h"
#include "Toy/Dialect.h"
#include "mlir/IR/Builders.h"
#include <cstdint>
#include <optional>

mlir::Location BackEnd::loc(const ast::Location &loc) {
  return mlir::FileLineColLoc::get(builder->getStringAttr(*loc.filename), loc.line, loc.col);
}
/// Declare a variable in the current scope, return success if the variable
/// wasn't declared yet.
llvm::LogicalResult BackEnd::declare(llvm::StringRef var, mlir::Value value) {
  if (symbolTable.count(var))
    return mlir::failure();
  symbolTable.insert(var, value);
  return mlir::success();
}

/// Build a tensor type from a list of shape dimensions.
mlir::Type BackEnd::getType(llvm::ArrayRef<int64_t> shape) {
  // If the shape is empty, then this type is unranked.
  if (shape.empty())
    return mlir::UnrankedTensorType::get(builder->getF64Type());

  // Otherwise, we use the given shape.
  return mlir::RankedTensorType::get(shape, builder->getF64Type());
}

mlir::Type BackEnd::getType(ast::Shape* type) { 
  auto dims = type->get_dims();
  auto shape = llvm::ArrayRef<int64_t>(dims); 
  return getType(shape); 
}

mlir::Type BackEnd::getType() {
  return getType(llvm::ArrayRef<int64_t>({}));
}

/// Build an MLIR type from a Toy AST variable type (forward to the generic
/// getType above).
void BackEnd::traverse() {
  for (auto f : this->ast->getFunctions()) {
    visitFunction(f);
  }
}

void BackEnd::visitFunction(ast::Function* func) {
  // Create a scope in the mlir symbol table for the variable declarations.
  // tbh, this had me confused until I remembered that this is how you can define
  // a variable implicitly using the type constructor. Bloody c++
  llvm::ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(this->symbolTable);

  // Create our function
  builder->setInsertionPointToEnd(this->module.getBody());
  auto location = loc(func->getLine());
  llvm::SmallVector<mlir::Type, 4> 
    argTypes(func->getParam().size(), getType());
  auto funcType = builder->getFunctionType(argTypes, std::nullopt);
  auto succ = builder->create<mlir::toy::FuncOp>(location, func->getName(), funcType);
}

void BackEnd::visitDecl(ast::Decl*) {}
void BackEnd::visitLiteralExpr(ast::LiteralExpr* ) {}
