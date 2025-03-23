#ifndef __TOY_AST_BUILDER_
#define __TOY_AST_BUILDER_

#include <AST/AST.h>
#include <csignal>
#include <cstddef>
#include <optional>
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

  protected:
    ast::Variable* define_variable(ast::QualifierType qualifier,
                                   ast::GazType type,
                                   std::string name,
                                   Token* token);
    void define_subroutine(ast::SubRoutine* subroutine,
                           std::string name,
                           Token* token);
    ast::Variable* resolve_variable(std::string name, Token* token);
    std::optional<ast::SubRoutine*> resolve_subroutine(std::string name);
    std::optional<Symbol*> resolve(std::string name);
    std::optional<Symbol*> resolve_local(std::string name);


  public:
    ASTBuilder();
    
    bool has_ast() { return ast ? true : false; };
    ast::Block* get_ast() { return ast; };
    std::any visitFile(ToyParser::FileContext *ctx) override;
    std::any visitStat(ToyParser::StatContext *ctx) override;
    std::any visitFuncStat(ToyParser::FuncStatContext *ctx) override;
    std::any visitBlockStat(ToyParser::BlockStatContext *ctx) override;
    std::any visitDeclStat(ToyParser::DeclStatContext *ctx) override;
    std::any visitShape(ToyParser::ShapeContext *ctx) override;
    std::any visitMulDivMod(ToyParser::MulDivModContext *ctx) override;
    std::any visitFunc(ToyParser::FuncContext *ctx) override;
    std::any visitBracket(ToyParser::BracketContext *ctx) override;
    std::any visitVariable(ToyParser::VariableContext *ctx) override;
    std::any visitAddSub(ToyParser::AddSubContext *ctx) override;
    std::any visitLiteralSlice(ToyParser::LiteralSliceContext *ctx) override;
    std::any visitLiteral(ToyParser::LiteralContext *ctx) override;
    std::any visitFuncExpr(ToyParser::FuncExprContext *ctx) override;
    std::any visitSlice(ToyParser::SliceContext *ctx) override;
    std::any visitParams(ToyParser::ParamsContext *ctx) override;
};

#endif
