#include "AST/ASTVisitor.h"
#include "AST/AST.h"
#include "macros.h"
#include "CompileTimeExceptions.h"
#include <memory>

namespace pass {
void ASTPass::visitModule(ast::Module* ast) {
  for (auto f : ast->getFunctions()) {
    visitFunction(f);
  }
}

void ASTPass::visitFunction(ast::Function* func) {
  for (auto param : func->getParam())
    visitParam(param);
  assert(func->getBlock().has_value());
  visitBlock(func->getBlock().value());
}

void ASTPass::visitBlock(ast::Block* block) {
  for (auto stat : block->getStatements()) {
    if (std::holds_alternative<ast::Decl*>(stat)) visitDecl(std::get<ast::Decl*>(stat));
    else if (std::holds_alternative<ast::Return*>(stat)) visitReturn(std::get<ast::Return*>(stat));
    else if (std::holds_alternative<ast::Print*>(stat)) visitCallExpr(std::get<ast::Print*>(stat));
    else throw StatementError(block->getLine(), "Statement is not valid in a block");
  }
}

void ASTPass::visitExpr(ast::expr_t stat) {
  if (std::holds_alternative<ast::BinExpr*>(stat)) visitBinExpr(std::get<ast::BinExpr*>(stat));
  else if (std::holds_alternative<ast::CallExpr*>(stat)) visitCallExpr(std::get<ast::CallExpr*>(stat));
  else if (std::holds_alternative<ast::VarExpr*>(stat)) visitVarExpr(std::get<ast::VarExpr*>(stat));
  else if (std::holds_alternative<ast::LiteralExpr*>(stat)) visitLiteralExpr(std::get<ast::LiteralExpr*>(stat));
  else throw StatementError(ast::builtinloc, "Invalid expression found");
}

void ASTPass::visitReturn(ast::Return* ret) {
  if (ret->getExpr().has_value())
    visitExpr(ret->getExpr().value());
}
void ASTPass::visitDecl(ast::Decl* dec) {
  if (dec->getExpr().has_value())
    visitExpr(dec->getExpr().value());
}
void ASTPass::visitCallExpr(ast::CallExpr* ex) {
  for (auto e : ex->getOperands()) {
    visitExpr(e);
  }
}
void ASTPass::visitBinExpr(ast::BinExpr* bin) {
  visitExpr(bin->getLHS());
  visitExpr(bin->getRHS());
}
void ASTPass::visitLiteralExpr(ast::LiteralExpr* lit) {
  if (std::holds_alternative<std::vector<double>>(lit->getValues())) return;

  auto vals = std::get<std::vector<ast::LiteralExpr*>>(lit->getValues());
  for (auto tensor : vals) {
    visitLiteralExpr(tensor);
  }
}
void ASTPass::visitVarExpr(ast::VarExpr* var) { return; }
void ASTPass::visitParam(ast::Param* parm) { return; }

}
