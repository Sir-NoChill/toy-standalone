#ifndef __TOY_SYMBOL_TABLE_
#define __TOY_SYMBOL_TABLE_

#include <stack>
#include <string>
#include <unordered_map>
#include <variant>

#include "AST/AST.h"

namespace ast {

class SymbolTable {
  std::unordered_map<std::string, ast::Function> symbolList;

  // need to initialize builtin symbols
  //  ex. print, transpose
  SymbolTable();
};
}


#endif
