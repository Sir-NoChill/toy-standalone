#ifndef __TOY_PASS_
#define __TOY_PASS_

#include "AST/AST.h"
#include "ASTPasses/TestPass.h"
#include "ASTPasses/Rewrite.h"

namespace pass {
inline void run_all_passes(ast::Module* ast) {
#ifdef DEBUG
  auto test = TestPass(ast);
  test.traverse();
#endif
  auto rewrite = Rewrite(ast);
  rewrite.traverse();
  // auto def = DefPass(ast);
  // def.traverse();
  // std::ofstream out = std::ofstream("sym.ast");
  // def.getSymbolTable()->dump(out);
}
}

#endif
