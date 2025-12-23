add_compile_options(-g)
add_executable(MyTest test_src.cpp)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
# get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)
get_property(deepgengraph_libs GLOBAL PROPERTY DEEPGENGRAPH_LIBS)

target_link_libraries(MyTest PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
#  ${triton_libs}
  ${deepgengraph_libs}

  # MLIR core
  MLIROptLib
  MLIRPass
  MLIRTransforms

  DeepgengraphIR
  DeepgengraphTritonIR
)
