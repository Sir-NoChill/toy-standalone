#ifndef __TOY_AST_
#define __TOY_AST_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <variant>
#include <fstream>
#include <vector>

namespace ast {

class BinExpr;
class CallExpr;
class VarExpr;
class LiteralExpr;
typedef std::variant<BinExpr*, CallExpr*, VarExpr*, LiteralExpr*> expr_t;

class Decl;
class Return;
class Print;
typedef std::variant<Decl*, Return*, Print*> stat_t;

class ASTNode {
  protected:
    uint16_t line;

    ASTNode() : line(0) {}
    ASTNode(uint16_t line) : line(line) {}

    virtual void dump(std::ofstream& outfile, int level) = 0;


    std::size_t loc();
    ~ASTNode(){};
};

class Shape : public ASTNode {
  private:
    std::vector<uint16_t> shape;
  public:
    Shape(uint16_t line, std::vector<uint16_t> shape)
      : ASTNode(line), shape(shape) {}
    Shape(Shape& other, uint16_t current=0) // badness (insert begin is O(n), shouldn't matter)
      : ASTNode(other.line) 
    { this->shape = other.shape; this->shape.insert(this->shape.begin(), current); } 

    void dump(std::ofstream& outfile, int level) override { }
    uint16_t num_values();
    std::vector<uint16_t> get_dims() { return shape; };
    std::string repr();
};

class Param : public ASTNode {
  private:
    std::optional<Shape*> shape;
    std::string name;
  public:
    Param(
	uint16_t line,
	std::optional<Shape*> shape, 
	std::string name
    ) : ASTNode(line), shape(shape), name(name) {}
    void dump(std::ofstream& outfile, int level) override;
    std::string getName() { return name; }
    std::optional<Shape*> getShape() { return shape; }
};

class Return : public ASTNode {
  private:
    std::optional<expr_t> expression;
  public:
    Return(uint16_t line, std::optional<expr_t> exp)
      : ASTNode(line), expression(exp) {}
    void dump(std::ofstream& outfile, int level) override;
};

class Decl : public ASTNode {
  private:
    Param* var;
    std::optional<expr_t> expr;
    
  public:
    Decl(uint16_t line, Param* var, std::optional<expr_t> expr)
      : ASTNode(line), var(var), expr(expr) {}
    void dump(std::ofstream& outfile, int level) override;
};

class Print : public ASTNode {
  private:
    std::optional<expr_t> expression;
  public:
    Print(uint16_t line, std::optional<expr_t> expr) 
      : ASTNode(line), expression(expr) {}
    void dump(std::ofstream& outfile, int level) override;
};

class Block : public ASTNode {
  private:
    std::vector<stat_t> statements;
  public:
    Block(uint16_t line, std::vector<stat_t> statements)
      : ASTNode(line), statements(statements) {}
    void dump(std::ofstream& outfile, int level) override;
};

class Function : public ASTNode {
  private:
    std::string prototype;
    std::optional<Block*> body;
    std::vector<Param*> parameters;
  public:
    Function(
	std::string proto, 
	uint64_t line, 
	std::optional<Block*> b, 
	std::vector<Param*> parm
    ) : ASTNode(line), prototype(proto), body(b), parameters(parm) {}
    void dump(std::ofstream& outfile, int level) override;
};

class Module : public ASTNode {
  private:
    std::vector<Function*> functions;
  public:
    Module(std::vector<Function*> functions);
    void dump(std::ofstream& outfile, int level) override;
};

class Expr : public ASTNode {
  protected:
    std::optional<Shape*> shape;
  public:
    Expr(uint16_t line, std::optional<Shape*> shape)
      : ASTNode(line), shape(shape) {}
    void dump(std::ofstream& outfile, int level) override = 0;
};

class VarExpr : public Expr {
  private:
    std::string name;
    std::optional<std::vector<double>> vals;
  public:
    VarExpr(
	uint16_t line, 
	std::string name,
	std::optional<Shape*> shape,
	std::optional<std::vector<double>> vals
    ) : Expr(line, shape), name(name), vals(vals) {}

    void dump(std::ofstream& outfile, int level) override;
};

class CallExpr : public Expr {
  private:
    std::string func;
    std::vector<expr_t> operands;
  public:
    CallExpr(uint16_t line, std::string name, std::vector<expr_t> operands)
      : Expr(line, std::nullopt), func(name), operands(operands) {}

    void dump(std::ofstream& outfile, int level) override;
};

enum Operation {
  Add,
  Sub,
  Mul,
  Div,
  MatMul,
};
std::string opToString(Operation op);

class BinExpr : public Expr {
  private:
    Operation op;
    expr_t lhs;
    expr_t rhs;
  public:
    BinExpr(
	uint16_t line,
	std::optional<Shape*> shape, 
	Operation op, 
	expr_t lhs, 
	expr_t rhs
    ) : Expr(line, shape), op(op), lhs(lhs), rhs(rhs) {}
    void dump(std::ofstream& outfile, int level) override;
};

class LiteralExpr;
typedef std::variant<std::vector<double>, std::vector<LiteralExpr*>> tensor_t;
class LiteralExpr : public Expr {
  private:
    tensor_t values;
  public:
    LiteralExpr(
      uint16_t line,
      std::optional<Shape*> shape,
      tensor_t values
    ) : Expr(line, shape), values(values) {}

    void dump(std::ofstream& outfile, int level) override;
    std::optional<Shape*> getShape() { return this->shape; }
    std::string repr();
    size_t size();
};

} // namespace ast

#endif
