#include <algorithm>
#include <cassert>
#include <cstdint>
#include <string>
#include <print>

#include "AST/AST.h"
#include "CompileTimeExceptions.h"

namespace ast {

void VarExpr::dump(std::ofstream& outfile, int level) {
  std::print(outfile, "{}Var {} @location {}: ",std::string(level, ' '), this->name, this->location.repr());
  if (this->shape.has_value()) {
    this->shape.value()->dump(outfile, 0);
  } else {
    outfile << "<Undef>";
  }

  if (this->vals.has_value()) {
    outfile << "[";
    for (auto i : this->vals.value()) {
      outfile << i << ", ";
    }
    outfile << "]";
  }
  outfile << std::endl;
}

void LiteralExpr::dump(std::ofstream& outfile, int level) {
  std::string indent = std::string(level, ' ');
  std::print(outfile, "{}Literal:\n{} {}\n",indent, indent , this->repr());
}

void CallExpr::dump(std::ofstream& outfile, int level) {
  std::print(outfile, "{}Call '{}' (def @ location {}) with args:\n",
      std::string(level, ' '),
      this->func,
      this->location.repr()
  );
  for (auto e : this->operands) 
    std::visit([&](auto& arg){arg->dump(outfile, level + 1);}, e);
}

Module::Module(std::vector<ast::Function*> f) : functions(f) {}

void Module::dump(std::ofstream& outfile, int level) {
  auto spaces = std::string(level, ' ');
  std::print(outfile, "{}Module:\n", spaces);
  for (auto func : functions) {
    func->dump(outfile, level + 1);
  }
}

std::string LiteralExpr::repr() {
  // assert(this->size() == this->shape.value()->num_values());
  auto lit = std::get_if<std::vector<LiteralExpr*>>(&this->values);
  auto val = std::get_if<std::vector<double>>(&this->values);

  std::string repr;

  if (lit) {
    repr.append(this->shape.has_value() ? this->shape.value()->repr() : "<Undef>");
    repr.append(" [");
    for (auto child : *lit) {
      repr.append(child->repr());
    }
    repr.append("]");
  }

  if (val) {
    repr.append("<");
    repr.append(std::to_string(val->size()));
    repr.append("> [");
    for (auto v : *val) {
      repr.append(std::to_string(v));
      repr.append(", ");
    }
    repr.append("]");
  }

  return repr;
}

void Function::dump(std::ofstream& outfile, int level) {
  auto spaces = std::string(level, ' ');

  std::print(outfile, "{}Function '{}' @location {} (", spaces, prototype, location.repr());
  int i = 0;
  for (auto param : this->parameters) {
    // coopting the level as the param index
    param->dump(outfile, i);
  }
  std::print(outfile, ") :\n");
  // body->dump(outfile, level + 1);
  if (this->body.has_value()) {
    this->body.value()->dump(outfile, level + 1);
  }
}

void Param::dump(std::ofstream& outfile, int index) {
  int i = 0;
  std::print(outfile, "{} {}: {}, ", 
      i, 
      this->shape.has_value() ? this->shape.value()->repr() : "<Undef>", 
      this->name
  );
}


void Print::dump(std::ofstream& outfile, int index) {
  std::print(outfile, "{}Print @location {}: {}\n", 
      std::string(index, ' '), this->location.repr(), "None");
}

void Block::dump(std::ofstream& outfile, int index) {
  std::print(outfile, "{}Block:\n", std::string(index, ' '));
  for (auto stat : statements) {
    std::visit([&](auto& arg){arg->dump(outfile, index + 1);}, stat);
  }
}

void Decl::dump(std::ofstream& outfile, int level) {
  std::print(outfile, "{}Decl {} {} @location {}:\n", 
      std::string(level, ' '),
      this->var->getName(),
      this->var->getShape().has_value() ? this->var->getShape().value()->repr() : "<Undef>",
      this->location.repr()
  );
  if (this->expr.has_value())
    std::visit([&](auto& arg){arg->dump(outfile, level + 1);}, this->expr.value());
}

void Return::dump(std::ofstream& outfile, int level) {
  std::print(outfile, "{}Return @location {}: \n", std::string(level, ' '), this->location.repr());
  if (this->expr.has_value())
    std::visit([&](auto& arg){arg->dump(outfile, level + 1);}, this->expr.value());
}

void BinExpr::dump(std::ofstream& outfile, int level) {
  std::print(outfile, "{}BinOp op={}:\n", 
      std::string(level, ' '),
      ast::opToString(this->op));
  std::visit([&](auto& arg){arg->dump(outfile, level + 1);}, this->lhs);
  std::visit([&](auto& arg){arg->dump(outfile, level + 1);}, this->rhs);
}

std::string opToString(Operation op) {
  switch (op) {
    case Add: return "+";
    case Sub: return "-";
    case Mul: return "*";
    case Div: return "/";
    case MatMul: return "**";
  }
  throw LiteralError(builtinloc, "Invalid operation found");
}

uint16_t Shape::num_values() {
  assert (shape.size() >= 2);
  uint16_t vals = 1;

  for (auto val : shape) {
    vals = vals * val;
  }

  return vals;
}

std::string Shape::repr() {
  std::string r = "<";
  for (auto i : this->shape) {
    r.append(std::to_string(i));
    r.append(", ");
  }
  r.append(">");
  return r;
}            

size_t LiteralExpr::size() {
  auto lit = std::get_if<std::vector<LiteralExpr*>>(&this->values);
  auto val = std::get_if<std::vector<double>>(&this->values);
  if (lit) {
    assert(!val);
    assert(lit->size() > 0);

    // we can rely on this to recurse down to the values
    //  and since a tensor is symetric, we should not have
    //  a size mismatch between elements of this system
    return lit->size() + lit->at(0)->size();
  }

  if (val) {
    assert(!lit);
    return val->size();
  }

  throw TypeError(builtinloc, "Failed to extract literal size (internal error)");
}
}
