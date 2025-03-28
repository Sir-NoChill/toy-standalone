#ifndef __TOY_ASTVISITOR_
#define __TOY_ASTVISITOR_
#include "AST/AST.h"

namespace pass {
class ASTPass {
  public:
    ast::Module* ast;
    // ast::SymbolTable* symtab;

    explicit ASTPass(ast::Module* ast) : ast(ast) {}

    virtual void traverse() = 0;
    
    virtual void visitModule(ast::Module*);
    virtual void visitFunction(ast::Function*);
    virtual void visitParam(ast::Param*);
    virtual void visitBlock(ast::Block*);
    virtual void visitReturn(ast::Return*);
    virtual void visitDecl(ast::Decl*);
    virtual void visitExpr(ast::expr_t);
    virtual void visitCallExpr(ast::CallExpr*);
    virtual void visitBinExpr(ast::BinExpr*);
    virtual void visitVarExpr(ast::VarExpr*);
    virtual void visitLiteralExpr(ast::LiteralExpr*);
};
}

#endif
