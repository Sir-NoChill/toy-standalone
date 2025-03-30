#include "AST/ASTVisitor.h"

namespace pass {
class TestPass : public ASTPass {
  public:
    TestPass(ast::Module* ast) : ASTPass(ast) {}
    void traverse() override { visitModule(this->ast); }
};
}

