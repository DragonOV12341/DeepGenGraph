module {
  func.func @Attn_p0(%arg0: tensor<1x4096x32x128xf16>, %arg1: tensor<1x4096x32x128xf16>, %arg2: tensor<1x4096x32x128xf16>, %arg3: tensor<1x32x4096x1xf32>) -> tensor<1x4096x32x128xf16> {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0xFC00 : f16
    %cst_0 = arith.constant 0.000000e+00 : f16
    %c1 = arith.constant 1 : index
    %cst_1 = arith.constant 1.131250e+01 : f16
    %0 = bufferization.to_memref %arg1 : tensor<1x4096x32x128xf16> to memref<1x4096x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>
    %1 = bufferization.to_memref %arg3 : tensor<1x32x4096x1xf32> to memref<1x32x4096x1xf32, strided<[?, ?, ?, ?], offset: ?>>
    %2 = bufferization.to_memref %arg2 : tensor<1x4096x32x128xf16> to memref<1x4096x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>
    %3 = bufferization.to_memref %arg0 : tensor<1x4096x32x128xf16> to memref<1x4096x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x32x4096x4096xf16>
    affine.parallel (%arg4) = (0) to (1) {
      affine.parallel (%arg5) = (0) to (32) {
        affine.parallel (%arg6) = (0) to (4096) {
          affine.for %arg7 = 0 to 4096 {
            affine.store %cst_0, %alloc[%arg4, %arg5, %arg6, %arg7] : memref<1x32x4096x4096xf16>
          }
        }
      }
    }
    affine.parallel (%arg4) = (0) to (1) {
      affine.parallel (%arg5) = (0) to (32) {
        affine.parallel (%arg6) = (0) to (4096) {
          affine.for %arg7 = 0 to 4096 {
            affine.for %arg8 = 0 to 128 {
              %5 = affine.load %3[%arg4, %arg6, %arg5, %arg8] : memref<1x4096x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>
              %6 = affine.load %2[%arg4, %arg7, %arg5, %arg8] : memref<1x4096x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>
              %7 = affine.load %alloc[%arg4, %arg5, %arg6, %arg7] : memref<1x32x4096x4096xf16>
              %8 = arith.mulf %5, %6 : f16
              %9 = arith.addf %7, %8 : f16
              affine.store %9, %alloc[%arg4, %arg5, %arg6, %arg7] : memref<1x32x4096x4096xf16>
            }
          }
        }
      }
    }
    %alloc_2 = memref.alloc() {alignment = 64 : i64} : memref<1x32x4096x128xf16>
    affine.parallel (%arg4) = (0) to (1) {
      affine.parallel (%arg5) = (0) to (32) {
        affine.parallel (%arg6) = (0) to (4096) {
          affine.for %arg7 = 0 to 128 {
            affine.store %cst_0, %alloc_2[%arg4, %arg5, %arg6, %arg7] : memref<1x32x4096x128xf16>
          }
        }
      }
    }
    affine.parallel (%arg4) = (0) to (1) {
      affine.parallel (%arg5) = (0) to (32) {
        affine.parallel (%arg6) = (0) to (4096) {
          affine.for %arg7 = 0 to 128 {
            affine.for %arg8 = 0 to 4096 {
              %5 = affine.load %alloc[%arg4, %arg5, %arg6, %arg8] : memref<1x32x4096x4096xf16>
              %6 = affine.load %1[%arg4, %arg5, %arg6, %c0] : memref<1x32x4096x1xf32, strided<[?, ?, ?, ?], offset: ?>>
              %7 = affine.load %0[%arg4, %arg8, %arg5, %arg7] : memref<1x4096x32x128xf16, strided<[?, ?, ?, ?], offset: ?>>
              %8 = affine.load %alloc_2[%arg4, %arg5, %arg6, %arg7] : memref<1x32x4096x128xf16>
              %9 = arith.addi %arg8, %c1 : index
              %10 = arith.cmpi sle, %arg6, %9 : index
              %11 = arith.select %10, %cst, %cst_0 : f16
              %12 = arith.divf %5, %cst_1 : f16
              %13 = arith.addf %12, %11 : f16
              %14 = arith.extf %13 : f16 to f32
              %15 = math.exp %14 : f32
              %16 = arith.divf %15, %6 : f32
              %17 = arith.truncf %16 : f32 to f16
              %18 = arith.mulf %17, %7 : f16
              %19 = arith.addf %8, %18 : f16
              affine.store %19, %alloc_2[%arg4, %arg5, %arg6, %arg7] : memref<1x32x4096x128xf16>
            }
          }
        }
      }
    }
    %alloc_3 = memref.alloc() {alignment = 64 : i64} : memref<1x4096x32x128xf16>
    affine.parallel (%arg4) = (0) to (1) {
      affine.parallel (%arg5) = (0) to (4096) {
        affine.parallel (%arg6) = (0) to (32) {
          affine.for %arg7 = 0 to 128 {
            %5 = affine.load %alloc_2[%arg4, %arg6, %arg5, %arg7] : memref<1x32x4096x128xf16>
            affine.store %5, %alloc_3[%arg4, %arg5, %arg6, %arg7] : memref<1x4096x32x128xf16>
          }
        }
      }
    }
    %4 = bufferization.to_tensor %alloc_3 : memref<1x4096x32x128xf16> to tensor<1x4096x32x128xf16>
    return %4 : tensor<1x4096x32x128xf16>
  }
}