#ifndef __TOY_AST_
#define __TOY_AST_

#include "Token.h"
#include <cstddef>

namespace ast {

class ASTNode {
  public:
    antlr4::Token* token;

    ASTNode();
    ASTNode(antlr4::Token* token);

    virtual void dump_dot(std::ofstream& outfile, int level) = 0;
    virtual void dump_xml(std::ofstream& outfile, int level) = 0;

    std::size_t loc();
    ~ASTNode(){};
};

class Function : public ASTNode {
  public:
    void dump_dot(std::ofstream& outfile, int level) override { }
    void dump_xml(std::ofstream& outfile, int level) override { }
};

class Expr : public ASTNode {
  public:
    void dump_dot(std::ofstream& outfile, int level) override { }
    void dump_xml(std::ofstream& outfile, int level) override { }
};

class Decl : public ASTNode {
  public:
    void dump_dot(std::ofstream& outfile, int level) override { }
    void dump_xml(std::ofstream& outfile, int level) override { }
};

class Shape : public ASTNode {
  public:
    void dump_dot(std::ofstream& outfile, int level) override { }
    void dump_xml(std::ofstream& outfile, int level) override { }
};

class Slice : public ASTNode {
  public:
    void dump_dot(std::ofstream& outfile, int level) override { }
    void dump_xml(std::ofstream& outfile, int level) override { }
};

class Param : public ASTNode {
  public:
    void dump_dot(std::ofstream& outfile, int level) override { }
    void dump_xml(std::ofstream& outfile, int level) override { }
};

class Block : public ASTNode {
  public:
    void dump_dot(std::ofstream& outfile, int level) override { }
    void dump_xml(std::ofstream& outfile, int level) override { }
};

} // namespace ast

#endif
