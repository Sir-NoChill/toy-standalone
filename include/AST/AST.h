#ifndef __TOY_AST_
#define __TOY_AST_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <sys/types.h>
#include <variant>
#include <fstream>
#include <vector>

namespace ast {

struct Location {
  std::shared_ptr<std::string> filename;
  size_t line;
  size_t col;

  std::string repr() { return *filename + ":" + std::to_string(line) + "," + std::to_string(col); }
};

const Location builtinloc = Location(std::make_shared<std::string>(""), 0, 0);

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
    Location location;

    ASTNode() : location(Location(nullptr, 0, 0)) {}
    ASTNode(Location location) : location(location) {}

    virtual void dump(std::ofstream& outfile, int level) = 0;

    ~ASTNode(){};

  public:
    Location getLine() { return location; }
};

class Shape : public ASTNode {
  private:
    std::vector<int64_t> shape;
  public:
    Shape(Location line, std::vector<int64_t> shape)
      : ASTNode(line), shape(shape) {}
    Shape(Shape& other, uint16_t current=0) // badness (insert begin is O(n), shouldn't matter)
      : ASTNode(other.location) 
    { this->shape = other.shape; this->shape.insert(this->shape.begin(), current); } 

    void dump(std::ofstream& outfile, int level) override { }
    uint16_t num_values();
    std::vector<int64_t> get_dims() { return shape; };
    std::string repr();
};

class Param : public ASTNode {
  private:
    std::optional<Shape*> shape;
    std::string name;
  public:
    Param(
	Location line,
	std::optional<Shape*> shape, 
	std::string name
    ) : ASTNode(line), shape(shape), name(name) {}
    Param(Param& other) 
      : ASTNode(other.location), shape(other.shape), name(other.name) {}

    void dump(std::ofstream& outfile, int level) override;
    std::string getName() { return name; }
    std::optional<Shape*> getShape() { return shape; }
};

class Return : public ASTNode {
  private:
    std::optional<expr_t> expr;
  public:
    Return(Location line, std::optional<expr_t> exp)
      : ASTNode(line), expr(exp) {}
    void dump(std::ofstream& outfile, int level) override;
    std::optional<expr_t> getExpr() { return expr; };
};

class Decl : public ASTNode {
  private:
    Param* var;
    std::optional<expr_t> expr;
    
  public:
    Decl(Location line, Param* var, std::optional<expr_t> expr)
      : ASTNode(line), var(var), expr(expr) {}
    void dump(std::ofstream& outfile, int level) override;
    std::optional<expr_t> getExpr() { return expr; };
    Param* getParam() { return var; }
};

class Block : public ASTNode {
  private:
    std::vector<stat_t> statements;
  public:
    Block(Location line, std::vector<stat_t> statements)
      : ASTNode(line), statements(statements) {}
    void dump(std::ofstream& outfile, int level) override;
    std::vector<stat_t> getStatements() { return statements; }
};

class Function : public ASTNode {
  private:
    std::string prototype;
    std::optional<Block*> body;
    std::vector<Param*> parameters;
  public:
    Function(
	std::string proto, 
	Location line, 
	std::optional<Block*> b, 
	std::vector<Param*> parm
    ) : ASTNode(line), prototype(proto), body(b), parameters(parm) {}
    void dump(std::ofstream& outfile, int level) override;

    std::vector<Param*> getParam() { return parameters; }
    std::string getName() { return prototype; }
    std::optional<Block*> getBlock() { return body; }
};

class Module : public ASTNode {
  private:
    std::vector<Function*> functions;
  public:
    Module(std::vector<Function*> functions);
    void dump(std::ofstream& outfile, int level) override;
    std::vector<Function*> getFunctions() { return functions; ;}
};

class Expr : public ASTNode {
  protected:
    std::optional<Shape*> shape;
  public:
    Expr(Location line, std::optional<Shape*> shape)
      : ASTNode(line), shape(shape) {}
    void dump(std::ofstream& outfile, int level) override = 0;
};

class VarExpr : public Expr {
  private:
    std::string name;
    std::optional<std::vector<double>> vals;
  public:
    VarExpr(
	Location line, 
	std::string name,
	std::optional<Shape*> shape,
	std::optional<std::vector<double>> vals
    ) : Expr(line, shape), name(name), vals(vals) {}

    void dump(std::ofstream& outfile, int level) override;
    void setShape(Shape* sh) { this->shape = sh; }
    void setLine(Location l) { this->location = l; }
    std::string getName() { return name; }
};

class CallExpr : public Expr {
  private:
    std::string func;
  protected:
    std::vector<expr_t> operands;
  public:
    CallExpr(Location line, std::string name, std::vector<expr_t> operands)
      : Expr(line, std::nullopt), func(name), operands(operands) {}

    void dump(std::ofstream& outfile, int level) override;
    std::vector<expr_t> getOperands() { return operands; }
    std::string getName() { return func; }
    void setLine(Location line) { this->location = line; }
};

class Print : public CallExpr {
  public:
    Print(Location line, std::optional<expr_t> expr) 
      : CallExpr(line, "print", std::vector<expr_t>()) {if (expr.has_value()) operands.emplace_back(expr.value());}
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
	Location line,
	std::optional<Shape*> shape, 
	Operation op, 
	expr_t lhs, 
	expr_t rhs
    ) : Expr(line, shape), op(op), lhs(lhs), rhs(rhs) {}
    void dump(std::ofstream& outfile, int level) override;

    expr_t getLHS() { return lhs; }
    expr_t getRHS() { return rhs; }
};

class LiteralExpr;
typedef std::variant<std::vector<double>, std::vector<LiteralExpr*>> tensor_t;
class LiteralExpr : public Expr {
  private:
    tensor_t values;
  public:
    LiteralExpr(
      Location line,
      std::optional<Shape*> shape,
      tensor_t values
    ) : Expr(line, shape), values(values) {}

    void dump(std::ofstream& outfile, int level) override;
    std::optional<Shape*> getShape() { return this->shape; }
    std::string repr();
    size_t size();

    tensor_t getValues() { return values; }
};

} // namespace ast

#endif
