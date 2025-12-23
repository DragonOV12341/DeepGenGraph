#ifndef DEEPGENGRAPHTODEEPGENGRAPHTRITON_CONVERSION_PASSES
#define DEEPGENGRAPHTODEEPGENGRAPHTRITON_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertDeepgengraphToLinalgOnTensor: Pass<"convert-deepgengraph-to-linalg-on-tensor", "::mlir::ModuleOp"> {
  let summary = "convert deepgengraph to linalg and tensor op";
  let description = [{}];

  let constructor = "mlir::deepgengraph::createConvertDeepgengraphToLinalgOnTensorPass()";

  let dependentDialects = [
    "mlir::deepgengraph::DeepgengraphDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect"
  ];
}



#endif // DEEPGENGRAPHTODEEPGENGRAPHTRITON_CONVERSION_PASSES