#ifndef __TOY_AST_BUILDER_
#define __TOY_AST_BUILDER_

#include <ast/ast.h>
#include "ToyBaseVisitor.h"

class ASTBuilder : public toy::ToyBaseVisitor {
  public:
    bool has_ast() { return false; };
    ast::Block* get_ast() { return nullptr; };
};

#endif
