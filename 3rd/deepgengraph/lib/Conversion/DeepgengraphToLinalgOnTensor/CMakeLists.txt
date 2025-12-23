add_deepgengraph_library(DeepgengraphToLinalgOnTensor
  DeepgengraphToLinalgOnTensorPass.cpp

  DEPENDS
  DeepgengraphToLinalgOnTensorPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  DeepgengraphIR 
)