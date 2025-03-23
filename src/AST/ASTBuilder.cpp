#include "AST/ASTBuilder.h"

using namespace toy;
ASTBuilder::ASTBuilder() 
  : ast_stack(std::vector<ast::ASTNode*>()), ast(nullptr), symtab(nullptr) {}

std::any ASTBuilder::visitFile(toy::ToyParser::FileContext *ctx) { return nullptr; }
std::any ASTBuilder::visitStat(ToyParser::StatContext *ctx) { return nullptr; }
std::any ASTBuilder::visitFuncStat(ToyParser::FuncStatContext *ctx) { return nullptr; }
std::any ASTBuilder::visitBlockStat(ToyParser::BlockStatContext *ctx) { return nullptr; }
std::any ASTBuilder::visitDeclStat(ToyParser::DeclStatContext *ctx) { return nullptr; } 
std::any ASTBuilder::visitShape(ToyParser::ShapeContext *ctx) { return nullptr; }
std::any ASTBuilder::visitMulDivMod(ToyParser::MulDivModContext *ctx) { return nullptr; }
std::any ASTBuilder::visitFunc(ToyParser::FuncContext *ctx) { return nullptr; }
std::any ASTBuilder::visitBracket(ToyParser::BracketContext *ctx) { return nullptr; }
std::any ASTBuilder::visitVariable(ToyParser::VariableContext *ctx) { return nullptr; }
std::any ASTBuilder::visitAddSub(ToyParser::AddSubContext *ctx) { return nullptr; }
std::any ASTBuilder::visitLiteralSlice(ToyParser::LiteralSliceContext *ctx) { return nullptr; }
std::any ASTBuilder::visitLiteral(ToyParser::LiteralContext *ctx) { return nullptr; }
std::any ASTBuilder::visitFuncExpr(ToyParser::FuncExprContext *ctx) { return nullptr; }
std::any ASTBuilder::visitSlice(ToyParser::SliceContext *ctx) { return nullptr; }
std::any ASTBuilder::visitParams(ToyParser::ParamsContext *ctx) { return nullptr; }
