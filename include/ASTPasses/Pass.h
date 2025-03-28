#ifndef __TOY_PASS_
#define __TOY_PASS_

#include "AST/AST.h"
#include "ASTPasses/TestPass.h"

namespace pass {
inline void run_all_passes(ast::Module* ast) {;
  auto test = TestPass(ast);
  test.traverse();
}
}

#endif
