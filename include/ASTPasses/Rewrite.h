#include "AST/ASTVisitor.h"
class Rewrite : pass::ASTPass {
  ast::Function* currentNode;
  std::vector<std::string> functions;
  public:
    Rewrite(ast::Module* ast) : ASTPass(ast), functions(std::vector<std::string>({"print", "transpose"})) {}

    void traverse() override;
    void visitFunction(ast::Function*) override;
    void visitVarExpr(ast::VarExpr *) override;
    void visitCallExpr(ast::CallExpr*) override;
};
