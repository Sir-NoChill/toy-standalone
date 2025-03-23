#include "CompileTimeExceptions.h"
#include "ToyLexer.h"
#include "ToyParser.h"

#include "ANTLRFileStream.h"
#include "CommonTokenStream.h"
#include "tree/ParseTree.h"
#include "tree/ParseTreeWalker.h"

#include "backend/BackEnd.h"
#include "ast/astBuilder.h"
#include "ast_passes/pass.h"

#include <iostream>
#include <fstream>

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cout << "Missing required argument.\n"
              << "Required arguments: <input file path> <output file path>\n";
    return 1;
  }
  std::ofstream outfile(argv[2]);

  // Open the file then parse and lex it.
  antlr4::ANTLRFileStream afs;
  afs.loadFromFile(argv[1]);
  toy::ToyLexer lexer(&afs);
  antlr4::CommonTokenStream tokens(&lexer);
  toy::ToyParser parser(&tokens);

  // Get the root of the parse tree. Use your base rule name.
  ASTBuilder ast_builder;
  try {
    // Generate our parse tree
    antlr4::tree::ParseTree *tree = parser.file();
    ast_builder.visit(tree);
    assert(ast_builder.has_ast());

    // Turn it into an ast
    pass::run_all_passes(ast_builder.get_ast());

    // Optionally generate the ast dump
    std::ofstream debugfile("/dev/stderr");
    ast_builder.get_ast()->dump_xml(debugfile, 0);

    // Generate the code
    BackEnd backend = BackEnd(ast_builder.get_ast());
    backend.codegen();
    backend.dumpLLVM(outfile);
  } catch (CompileTimeException const& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }

  // HOW TO USE A VISITOR
  // Make the visitor
  // MyVisitor visitor;
  // Visit the tree
  // visitor.visit(tree);

  return 0;
}
