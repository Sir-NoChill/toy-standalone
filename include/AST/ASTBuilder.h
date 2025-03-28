#ifndef __TOY_AST_BUILDER_
#define __TOY_AST_BUILDER_

#include <AST/AST.h>
#include <csignal>
#include <cstddef>
#include <optional>

#include "ToyBaseVisitor.h"
#include "ToyParser.h"

#include "Symbol/SymbolTable.h"
#include "CompileTimeExceptions.h"


using namespace toy;

class ASTBuilder : public ToyBaseVisitor {
  private:
    std::vector<ast::ASTNode*> ast_stack;
    ast::Module* ast;
    ast::SymbolTable* symtab;

  protected:
    ast::Param* defineVariable(
	ast::Shape shape,
        std::string name,
        antlr4::Token* token
    );

    void defineFunction(
	ast::Function* subroutine,
        std::string name,
        antlr4::Token* token
    );

    ast::Param* resolveVariable(std::string name, antlr4::Token* token);
    std::optional<ast::Function*> resolve_subroutine(std::string name);
    std::optional<ast::Function*> resolve(std::string name);
    std::optional<ast::Function*> resolve_local(std::string name);


  public:
    ASTBuilder();
    
    bool has_ast() { return ast ? true : false; };
    ast::Module* get_ast() { return ast; };
    std::any visitFile(ToyParser::FileContext *ctx) override;
    std::any visitStat(ToyParser::StatContext *ctx) override;
    std::any visitFuncStat(ToyParser::FuncStatContext *ctx) override;
    std::any visitBlockStat(ToyParser::BlockStatContext *ctx) override;
    std::any visitDeclStat(ToyParser::DeclStatContext *ctx) override;
    std::any visitReturnStat(ToyParser::ReturnStatContext *ctx) override;
    std::any visitShape(ToyParser::ShapeContext *ctx) override;
    std::any visitMulDivMatmul(ToyParser::MulDivMatmulContext *ctx) override;
    std::any visitCall(ToyParser::CallContext *ctx) override;
    std::any visitBracket(ToyParser::BracketContext *ctx) override;
    std::any visitVariable(ToyParser::VariableContext *ctx) override;
    std::any visitAddSub(ToyParser::AddSubContext *ctx) override;
    std::any visitLiteralSlice(ToyParser::LiteralSliceContext *ctx) override;
    std::any visitLiteral(ToyParser::LiteralContext *ctx) override;
    std::any visitFuncExpr(ToyParser::FuncExprContext *ctx) override;
    std::any visitParams(ToyParser::ParamsContext *ctx) override;
    std::any visitNested(ToyParser::NestedContext *ctx) override;
    std::any visitPrintStat(ToyParser::PrintStatContext *ctx) override;
    std::any visitSliceValue(ToyParser::SliceValueContext *ctx) override;
};

#endif
