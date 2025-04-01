#ifndef __TOY_BACKEND_
#define __TOY_BACKEND_

// Pass manager
#include "AST/ASTVisitor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"

// Translation
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_os_ostream.h"

// LLVM Utilities
#include "llvm/ADT/ScopedHashTable.h"

// MLIR IR
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Verifier.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinOps.h"

// Dialects 
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

// Our dialect
#include "Toy/Dialect.h"

// AST
#include "AST/AST.h"

class BackEnd : public pass::ASTPass {
 public:
    BackEnd(ast::Module*);

    void codegen() { };
    void dumpLLVM(std::ostream &os);
    void dumpMLIR();
 
    // Pass
    void traverse() override;
    void visitFunction(ast::Function*) override;
    void visitDecl(ast::Decl*) override;
    void visitLiteralExpr(ast::LiteralExpr*) override;

 protected:
    void setupPrintf();
    void createGlobalString(const char *str, const char *stringName);
 
    int emitModule();
    int lowerDialects();

    llvm::LogicalResult declare(llvm::StringRef, mlir::Value);
    mlir::Location loc(const ast::Location&);
    mlir::Type getType(ast::Shape* shape);
    mlir::Type getType(llvm::ArrayRef<int64_t> shape);
    mlir::Type getType();
     
 private:
    // MLIR
    mlir::MLIRContext context;
    mlir::ModuleOp module;
    std::shared_ptr<mlir::OpBuilder> builder;

    // LLVM 
    llvm::LLVMContext llvm_context;
    std::unique_ptr<llvm::Module> llvm_module;
    llvm::ScopedHashTable<llvm::StringRef, mlir::Value> symbolTable;
};

#endif
