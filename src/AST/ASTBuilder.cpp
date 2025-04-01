#include "AST/ASTBuilder.h"
#include "AST/AST.h"
#include "CompileTimeExceptions.h"
#include "ToyParser.h"
#include "mlir/IR/Operation.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <sys/types.h>
#include <type_traits>

#define nodeCast(node, type) std::any_cast<type>(node) 

using namespace toy;
ASTBuilder::ASTBuilder() 
  : ast(nullptr), symtab(nullptr) {
}

ast::Location getLoc(antlr4::ParserRuleContext *ctx) {
  auto file = std::make_shared<std::string>(ctx->getStart()->getTokenSource()->getSourceName());
  auto line = ctx->getStart()->getLine();
  auto col = ctx->getStart()->getCharPositionInLine();

  return ast::Location(file, line, col);
}

std::any ASTBuilder::visitFile(toy::ToyParser::FileContext *ctx) {
  std::vector<ast::Function*> functions;
  for (auto child : ctx->stat()) {
    if (!child->funcStat()) 
      throw GlobalError(
	  getLoc(ctx),
	  "Only function definitions allowed in global scope"
      );
    ast::Function* func = nodeCast(visitFuncStat(child->funcStat()), ast::Function*);
    functions.emplace_back(func);
  }
  ast::Module* mod = new ast::Module(functions);

  ast_stack.emplace_back(std::move(mod));
  ast = mod;

  return nullptr;
}

std::any ASTBuilder::visitStat(ToyParser::StatContext *ctx) {
  // if (ctx->declStat() != nullptr) {

  if (ctx->printStat() != nullptr)
    return visitPrintStat(ctx->printStat());
  else if (ctx->declStat() != nullptr)
    return visitDeclStat(ctx->declStat());
  else if (ctx->returnStat() != nullptr)
    return visitReturnStat(ctx->returnStat());
  else
    throw StatementError(
	getLoc(ctx),
	"This method should only be called within a block, not on one"
    );
}

std::any ASTBuilder::visitFuncStat(ToyParser::FuncStatContext *ctx) {
  std::string name = ctx->ID()->getText();
  ast::Location loc = getLoc(ctx);
  std::vector<ast::Param*> params;
  if (ctx->params()) {
    params = nodeCast(visit(ctx->params()), std::vector<ast::Param*>);
  }

  std::optional<ast::Block*> block;
  if (ctx->blockStat()) {
    auto blk = nodeCast(visit(ctx->blockStat()), ast::Block*);
    block = std::make_optional(blk);
  }

  ast::Function* func = new ast::Function(name, loc, block, params);
  return func;
}

std::any ASTBuilder::visitParams(ToyParser::ParamsContext *ctx) {
  std::vector<ast::Param*> params;
  for (auto param : ctx->ID()) {
    auto var = new ast::Param(
	getLoc(ctx),
	std::nullopt, 
	param->toString()
    );
    params.emplace_back(std::move(var));
  }
  return params;
}

std::any ASTBuilder::visitBlockStat(ToyParser::BlockStatContext *ctx) {
  std::vector<
    std::variant<ast::Decl*, ast::Return*, ast::Print*>
    > statements;

  for (auto child : ctx->stat()) {
    std::any node = visit(child);
    if (node.type() == typeid(ast::Print*)) {
      statements.emplace_back(nodeCast(node, ast::Print*));
    } else if (node.type() == typeid(ast::Decl*)) {
      statements.emplace_back(nodeCast(node, ast::Decl*));
    } else if (node.type() == typeid(ast::Return*)) {
      statements.emplace_back(nodeCast(node, ast::Return*));
    } else {
      throw StatementError(getLoc(ctx), 
	  "Cannot have anything other than {decl, print, return} in block");
    }
  }
  return new ast::Block(getLoc(ctx), statements);
}

ast::expr_t exprCast(std::any node) {
  if (node.type() == typeid(ast::LiteralExpr*)) return nodeCast(node, ast::LiteralExpr*);
  if (node.type() == typeid(ast::CallExpr*)) return nodeCast(node, ast::CallExpr*);
  if (node.type() == typeid(ast::BinExpr*)) return nodeCast(node, ast::BinExpr*);
  if (node.type() == typeid(ast::VarExpr*)) return nodeCast(node, ast::VarExpr*);
  throw LiteralError(ast::builtinloc, "Expression contains unexpected type");
}
std::any ASTBuilder::visitDeclStat(ToyParser::DeclStatContext *ctx) {
  std::optional<ast::Shape*> shape;
  if (ctx->shape() != nullptr)
    shape = nodeCast(visit(ctx->shape()), ast::Shape*);

  std::string name = ctx->ID()->getText();
  auto parm = new ast::Param(getLoc(ctx), shape, name);
  ast::expr_t dec = exprCast(visit(ctx->expr()));

  return new ast::Decl(
      getLoc(ctx),
      parm,
      dec
  );
} 

std::any ASTBuilder::visitPrintStat(ToyParser::PrintStatContext *ctx) {
  return new ast::Print(getLoc(ctx), std::nullopt);
}

std::any ASTBuilder::visitReturnStat(ToyParser::ReturnStatContext *ctx) {
  std::optional<ast::expr_t> expr;
  if (ctx->expr())
    expr = exprCast(visit(ctx->expr()));
  return new ast::Return(getLoc(ctx), exprCast(visit(ctx->expr())));
}

std::any ASTBuilder::visitShape(ToyParser::ShapeContext *ctx) {
  std::vector<int64_t> dims;
  for (auto i : ctx->INT()) {
    dims.emplace_back(std::stoi(i->getText()));
  }
  return new ast::Shape(getLoc(ctx), dims);
}

ast::Operation getOp(std::string o) {
  if (o == "+") return ast::Operation::Add;
  if (o == "-") return ast::Operation::Sub;
  if (o == "*") return ast::Operation::Mul;
  if (o == "**") return ast::Operation::MatMul;
  if (o == "/") return ast::Operation::Div;
  throw LiteralError(ast::builtinloc, "Bad operation");
}
std::any ASTBuilder::visitAddSub(ToyParser::AddSubContext *ctx) {
  ast::Operation op = getOp(ctx->op->getText());

  assert(ctx->expr().size() == 2);
  auto lhs = exprCast(visit(ctx->expr(0)));
  auto rhs = exprCast(visit(ctx->expr(1)));

  return new ast::BinExpr(
      getLoc(ctx),
    std::nullopt,
    op, lhs, rhs
  );
}
std::any ASTBuilder::visitMulDivMatmul(ToyParser::MulDivMatmulContext *ctx) {
  ast::Operation op = getOp(ctx->op->getText());

  assert(ctx->expr().size() == 2);
  auto lhs = exprCast(visit(ctx->expr(0)));
  auto rhs = exprCast(visit(ctx->expr(1)));

  return new ast::BinExpr(
      getLoc(ctx),
    std::nullopt,
    op, lhs, rhs
  );
}

std::any ASTBuilder::visitLiteralSlice(ToyParser::LiteralSliceContext *ctx) {
  return visit(ctx->slice());
}
std::any ASTBuilder::visitNested(ToyParser::NestedContext *ctx) {
  std::vector<ast::LiteralExpr*> nested;
  for (auto t : ctx->slice()) {
    ast::LiteralExpr* nest = nodeCast(visit(t), ast::LiteralExpr*);
    nested.emplace_back(nest);
  }

  ast::Shape* sh = new ast::Shape(*nested.at(0)->getShape().value(), (uint16_t) nested.size());

  return new ast::LiteralExpr(getLoc(ctx), sh, nested);
}

std::any ASTBuilder::visitLiteral(ToyParser::LiteralContext *ctx) {
  auto vals = std::vector<double>();
  for (auto val : ctx->sliceValue()) {
    double v = nodeCast(visit(val), double);
    vals.emplace_back(v);
  }

  std::vector<int64_t> shvec;
  shvec.emplace_back(vals.size());
  auto shape = new ast::Shape(getLoc(ctx), shvec);

  return new ast::LiteralExpr(getLoc(ctx), shape, vals);
}

std::any ASTBuilder::visitSliceValue(ToyParser::SliceValueContext *ctx) {
  if (ctx->INT()) {
    return (double) std::stoi(ctx->INT()->getText());
  } else {
    return (double) std::stof(ctx->FLOAT()->getText());
  }
}

std::any ASTBuilder::visitCall(ToyParser::CallContext *ctx) {
  return visit(ctx->funcExpr());
}
std::any ASTBuilder::visitFuncExpr(ToyParser::FuncExprContext *ctx) {
  auto name = ctx->ID()->getText();
  std::vector<ast::expr_t> expr;
  for (auto e : ctx->expr()) {
    auto ex = exprCast(visit(e));
    expr.emplace_back(ex);
  }

  return new ast::CallExpr(getLoc(ctx), name, expr);
}

std::any ASTBuilder::visitBracket(ToyParser::BracketContext *ctx) { 
  if (not ctx->expr()) throw StatementError(getLoc(ctx), 
      "Empty brackets are not allowed as an expression");
  return visit(ctx->expr()); 
}
std::any ASTBuilder::visitVariable(ToyParser::VariableContext *ctx) { 
  return new ast::VarExpr(
      getLoc(ctx),
      ctx->ID()->getText(),
      std::nullopt,
      std::nullopt
  );
}
