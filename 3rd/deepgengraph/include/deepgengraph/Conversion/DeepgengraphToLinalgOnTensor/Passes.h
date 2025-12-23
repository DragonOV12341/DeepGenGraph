#ifndef DEEPGENGRAPH_CONVERSION_DEEPGENGRAPH_TO_LINALG_ON_TENSOR_PASS_H
#define DEEPGENGRAPH_CONVERSION_DEEPGENGRAPH_TO_LINALG_ON_TENSOR_PASS_H

#include "mlir/Pass/Pass.h"

namespace mlir::deepgengraph {

std::unique_ptr<mlir::Pass> createConvertDeepgengraphToLinalgOnTensorPass();

#define GEN_PASS_REGISTRATION
#include "deepgengraph/Conversion/DeepgengraphToLinalgOnTensor/Passes.h.inc"

} // namespace mlir::deepgengraph

#endif