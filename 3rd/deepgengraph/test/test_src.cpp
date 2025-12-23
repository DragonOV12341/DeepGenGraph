#include "deepgengraph/Dialect/Deepgengraph/IR/DeepgengraphDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/Support/raw_ostream.h"
#include "deepgengraph/Conversion/DeepgengraphToLinalgOnTensor/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/InitAllExtensions.h"

#include "mlir/InitAllDialects.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/InitAllPasses.h"

using namespace mlir;

int main(int argc, char ** argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllExtensions(registry);
  mlir::registerAllDialects(registry);
  auto ctx = std::make_unique<mlir::MLIRContext>(registry);

  // 首先，注册需要的 dialect
  ctx->loadDialect<
    func::FuncDialect, 
    arith::ArithDialect,
    tensor::TensorDialect,
    linalg::LinalgDialect,
    scf::SCFDialect,
    affine::AffineDialect,
    math::MathDialect,
    deepgengraph::DeepgengraphDialect
    >();

  
  // 读入文件
  auto src = parseSourceFile<ModuleOp>(argv[1], ctx.get());
  // 简单的输出，在 debug 的时候常用
  src->dump();

  mlir::PassManager pm(ctx.get());

  // pm.addNestedPass<deepgengraph::KernelOp>(deepgengraph::createConvertDeepgengraphToLinalgOnTensorPass());
  pm.addPass(deepgengraph::createConvertDeepgengraphToLinalgOnTensorPass());
  pm.addPass( mlir::createLinalgGeneralizeNamedOpsPass());
  pm.addPass( mlir::bufferization::createEmptyTensorEliminationPass()) ;
  pm.addPass(mlir::createLinalgElementwiseOpFusionPass());
  pm.run(src->getOperation());
  llvm::outs() << "\n====== after lower to linalg on tensor =====\n" ; llvm::outs().flush();
  src->dump();
  
  llvm::outs() << "\n====== after lower to affine =====\n" ; llvm::outs().flush();

  pm.addPass(mlir::bufferization::createOneShotBufferizePass());
  pm.addPass(mlir::createConvertTensorToLinalgPass());
  pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());

  mlir::affine::AffineParallelizeOptions opt;
  opt.maxNested = 3;  // bz.bx.by
  pm.addNestedPass<func::FuncOp>(mlir::affine::createAffineParallelize(opt));
  // pm.addPass(mlir::createConvertLinalgToParallelLoopsPass());  // to scf.parallel

  pm.addPass(mlir::affine::createLoopFusionPass(0,0,true,affine::FusionMode::Greedy)) ;
  
  pm.addPass(mlir::createLowerAffinePass()) ;
  pm.addPass(createParallelLoopFusionPass());


  
  // pm.addNestedPass<func::FuncOp>(mlir::affine::createLoopTilingPass(255*4)); 
  
  pm.run(src->getOperation());

  src->dump();
  return 0;
}
