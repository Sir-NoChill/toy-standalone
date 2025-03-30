#include "ASTPasses/Rewrite.h"
#include "AST/AST.h"
#include <variant>

void Rewrite::traverse() {
  for (auto func : this->ast->getFunctions()) {
    visitFunction(func);
  }
}

void Rewrite::visitFunction(ast::Function* func) {
  this->functions.emplace_back(func->getName());
  this->currentNode = func; // no nested functions or scopes, but hecking jank
  visitBlock(func->getBlock().value());
}

void Rewrite::visitVarExpr(ast::VarExpr* var) {
  for (auto p : this->currentNode->getParam()) {
    if (p->getName() == var->getName()) {
      var->setLine(this->currentNode->getLine());
    }
  }

  for (auto s : this->currentNode->getBlock().value()->getStatements()) {
    if (std::holds_alternative<ast::Decl*>(s)) {
      auto stat = std::get<ast::Decl*>(s);
      if (stat->getParam()->getName() == var->getName()) var->setLine(stat->getLine());
    }
  }
}

void Rewrite::visitCallExpr(ast::CallExpr* call) {
  if (call->getName() == "print" || call->getName() == "transpose")
    call->setLine(0);
  for (auto func : this->ast->getFunctions()) {
    if (func->getName() == call->getName()) {
      call->setLine(func->getLine());
    }
  }

  for (auto p : call->getOperands()) {
    visitExpr(p);
  }
}
