#ifndef __TOY_AST_BUILDER_
#define __TOY_AST_BUILDER_

#include <AST/AST.h>
#include <csignal>
#include <cstddef>
#include "ToyBaseVisitor.h"
#include "ToyParser.h"
#include "Symbol/SymbolTable.h"

using namespace toy;

struct Shape {
  std::vector<size_t> shape;
};

class ASTBuilder : public ToyBaseVisitor {
  private:
    std::vector<ast::ASTNode*> ast_stack;
    ast::Block* ast;
    ast::SymbolTable* symtab;

  public:
    ASTBuilder() : ast_stack(std::vector<ast::ASTNode*>()), ast(nullptr), symtab(nullptr) {}
    

    bool has_ast() { return false; };
    ast::Block* get_ast() { return nullptr; };
    std::any visitFile(ToyParser::FileContext *ctx) override { return nullptr; }
    std::any visitStat(ToyParser::StatContext *ctx) override { return nullptr; }
    std::any visitFuncStat(ToyParser::FuncStatContext *ctx) override { return nullptr; }
    std::any visitBlockStat(ToyParser::BlockStatContext *ctx) override { return nullptr; }
    std::any visitDeclStat(ToyParser::DeclStatContext *ctx) override { return nullptr; }
    std::any visitShape(ToyParser::ShapeContext *ctx) override { return nullptr; }
    std::any visitMulDivMod(ToyParser::MulDivModContext *ctx) override { return nullptr; }
    std::any visitFunc(ToyParser::FuncContext *ctx) override { return nullptr; }
    std::any visitBracket(ToyParser::BracketContext *ctx) override { return nullptr; }
    std::any visitVariable(ToyParser::VariableContext *ctx) override { return nullptr; }
    std::any visitAddSub(ToyParser::AddSubContext *ctx) override { return nullptr; }
    std::any visitLiteralSlice(ToyParser::LiteralSliceContext *ctx) override { return nullptr; }
    std::any visitLiteral(ToyParser::LiteralContext *ctx) override { return nullptr; }
    std::any visitFuncExpr(ToyParser::FuncExprContext *ctx) override { return nullptr; }
    std::any visitSlice(ToyParser::SliceContext *ctx) override { return nullptr; }
    std::any visitParams(ToyParser::ParamsContext *ctx) override { return nullptr; }
};

#endif
