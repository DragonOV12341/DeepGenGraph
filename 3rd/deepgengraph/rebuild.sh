#!/bin/bash
set -x
PROJECT_FOLDER=$(dirname $(readlink -f "$0"))
BUILD=${PROJECT_FOLDER}/build

MLIR_DIR=/home/xushilong/rocm-llvm-install/lib/cmake/mlir 
LLVM_BUILD=/home/xushilong/rocm-llvm-project/build
export CMAKE_PREFIX_PATH=$MLIR_DIR:$CMAKE_PREFIX_PATH

# triton: https://github.com/triton-lang/triton/pull/3325

# cmake .. -GNinja \
#   -DCMAKE_BUILD_TYPE=Release \
#   -DLLVM_ENABLE_ASSERTIONS=ON \
#   -DSTABLEHLO_ENABLE_BINDINGS_PYTHON=OFF \

#   -DTRITON_BUILD_PYTHON_MODULE=ON \
#   -DPython3_EXECUTABLE:FILEPATH=/home/zhongrx/miniconda3/envs/deepgengraph/bin/python \
#   -DCMAKE_VERBOSE_MAKEFILE:BOOL=ON \
#   -DPYTHON_INCLUDE_DIRS=/home/zhongrx/miniconda3/envs/deepgengraph/include/python3.10 \
#   -DPYBIND11_INCLUDE_DIR=/home/zhongrx/miniconda3/envs/deepgengraph/lib/python3.10/site-packages/pybind11/include \
#   -DTRITON_CODEGEN_BACKENDS="nvidia;amd" \
#   -DCUPTI_INCLUDE_DIR=/home/zhongrx/llm/tune/deepgengraph/3rd/triton/third_party/nvidia/backend/include \
#   -DTRITON_BUILD_PROTON=OFF \
#   -DMLIR_DIR=${MLIR_DIR}

BUILD_TYPE=Debug

# without python
# ref: https://github.com/triton-lang/triton/pull/3325
if [ -n "$1" ]; then
  rm -rf ${BUILD}
  mkdir ${BUILD} && cd ${BUILD}
  cmake ..  \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
    -DCMAKE_C_COMPILER=/usr/bin/gcc \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DSTABLEHLO_ENABLE_BINDINGS_PYTHON=OFF \
    -DTRITON_BUILD_PYTHON_MODULE=OFF \
    -DTRITON_CODEGEN_BACKENDS="nvidia;amd" \
    -DMLIR_DIR=/home/xushilong/rocm-llvm-install/lib/cmake/mlir 
else
  cd ${BUILD}
fi

cmake --build . -j64