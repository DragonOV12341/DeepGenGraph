// optimize Attn_p1
// -------- enter optimize
 deepgengraph.kernel @Attn_p1(%arg0: tensor<1x4096x32x128xf16> loc(unknown), %arg1: tensor<1x4096x32x128xf16> loc(unknown)) -> tensor<1x32x4096x1xf32> {
  %cst = arith.constant dense<1.131250e+01> : tensor<1xf16> loc(unknown)
  %0 = deepgengraph.trilu diagonal = 1, is_upper = true, shape = [4096, 4096], val = 0xFC00 : f16 loc(unknown)
  %1 = deepgengraph.permute %arg0, dims = [0, 2, 1, 3] : (tensor<1x4096x32x128xf16>) -> tensor<1x32x4096x128xf16> loc(unknown)
  %2 = deepgengraph.permute %arg1, dims = [0, 2, 3, 1] : (tensor<1x4096x32x128xf16>) -> tensor<1x32x128x4096xf16> loc(unknown)
  %3 = deepgengraph.dot %1, %2 : (tensor<1x32x4096x128xf16>, tensor<1x32x128x4096xf16>) -> tensor<1x32x4096x4096xf16> loc(unknown)
  %4 = deepgengraph.div %3, %cst : (tensor<1x32x4096x4096xf16>, tensor<1xf16>) -> tensor<1x32x4096x4096xf16> loc(unknown)
  %5 = deepgengraph.add %4, %0 : (tensor<1x32x4096x4096xf16>, tensor<4096x4096xf16>) -> tensor<1x32x4096x4096xf16> loc(unknown)
  %6 = deepgengraph.convert %5, type = f32 : (tensor<1x32x4096x4096xf16>) -> tensor<1x32x4096x4096xf32> loc(unknown)
  %7 = deepgengraph.exp %6 : (tensor<1x32x4096x4096xf32>) -> tensor<1x32x4096x4096xf32> loc(unknown)
  %8 = deepgengraph.reduce(%7), dim = -1, op =  ADD, keep_dim = true : (tensor<1x32x4096x4096xf32>) -> tensor<1x32x4096x1xf32> loc(unknown)
  deepgengraph.return %8 : tensor<1x32x4096x1xf32> loc(unknown)
} loc(unknown)

// -------- after add_deepgengraph_to_mask
 deepgengraph.kernel @Attn_p1(%arg0: tensor<1x4096x32x128xf64> {deepgengraph.erased_type = f16} loc(unknown), %arg1: tensor<1x4096x32x128xf64> {deepgengraph.erased_type = f16} loc(unknown)) -> (tensor<1x32x4096x1xf64> {deepgengraph.erased_type = f32}) {
  %cst = arith.constant 0.000000e+00 : f64 loc(unknown)
  %cst_0 = arith.constant 0xFFF0000000000000 : f64 loc(unknown)
  %c1 = arith.constant 1 : index loc(unknown)
  %cst_1 = arith.constant dense<0.12753105163574219> : tensor<1xf64> loc(unknown)
  %c0 = arith.constant 0 : index loc(unknown)
  %0 = deepgengraph.mask starts = [%c0, %c0], sizes = [4096, 4096], type = f64 {
  ^bb0(%arg2: index loc(unknown), %arg3: index loc(unknown)):
    %8 = arith.addi %arg2, %c1 : index loc(unknown)
    %9 = arith.cmpi ule, %8, %arg3 : index loc(unknown)
    %10 = scf.if %9 -> (f64) {
      scf.yield %cst_0 : f64 loc(unknown)
    } else {
      scf.yield %cst : f64 loc(unknown)
    } loc(unknown)
    deepgengraph.mask_yield %10 : f64 loc(unknown)
  } : (index, index) -> tensor<4096x4096xf64> loc(unknown)
  %1 = deepgengraph.permute %arg0, dims = [0, 2, 1, 3] : (tensor<1x4096x32x128xf64>) -> tensor<1x32x4096x128xf64> loc(unknown)
  %2 = deepgengraph.permute %arg1, dims = [0, 2, 3, 1] : (tensor<1x4096x32x128xf64>) -> tensor<1x32x128x4096xf64> loc(unknown)
  %3 = deepgengraph.mul %1, %cst_1 : (tensor<1x32x4096x128xf64>, tensor<1xf64>) -> tensor<1x32x4096x128xf64> loc(unknown)
  %4 = deepgengraph.dot %3, %2 : (tensor<1x32x4096x128xf64>, tensor<1x32x128x4096xf64>) -> tensor<1x32x4096x4096xf64> loc(unknown)
  %5 = deepgengraph.add %4, %0 : (tensor<1x32x4096x4096xf64>, tensor<4096x4096xf64>) -> tensor<1x32x4096x4096xf64> loc(unknown)
  %6 = deepgengraph.exp2 %5 : (tensor<1x32x4096x4096xf64>) -> tensor<1x32x4096x4096xf64> loc(unknown)
  %7 = deepgengraph.reduce(%6), dim = -1, op =  ADD, keep_dim = true : (tensor<1x32x4096x4096xf64>) -> tensor<1x32x4096x1xf64> loc(unknown)
  deepgengraph.return %7 : tensor<1x32x4096x1xf64> loc(unknown)
} loc(unknown)

// -------- after add_deepgengraph_parallelize
 deepgengraph.kernel @Attn_p1(%arg0: tensor<1x4096x32x128xf64> {deepgengraph.erased_type = f16} loc(unknown), %arg1: tensor<1x4096x32x128xf64> {deepgengraph.erased_type = f16} loc(unknown)) -> (tensor<1x32x4096x1xf64> {deepgengraph.erased_type = f32}) attributes {parallel_map = [{arg_dims = [0, 0], res_dims = [0], size_per_unit = 1 : i64, unit_num = 1 : i64}, {arg_dims = [1, -1], res_dims = [2], size_per_unit = 32 : i64, unit_num = 128 : i64}, {arg_dims = [2, 2], res_dims = [1], size_per_unit = 1 : i64, unit_num = 32 : i64}]} {
  %0 = deepgengraph.parallel_for %arg0, %arg1 {
  ^bb0(%arg2: index loc(unknown), %arg3: index loc(unknown), %arg4: index loc(unknown), %arg5: tensor<32x128xf64> loc(unknown), %arg6: tensor<4096x128xf64> loc(unknown)):
    %cst = arith.constant 0.000000e+00 : f64 loc(unknown)
    %cst_0 = arith.constant 0xFFF0000000000000 : f64 loc(unknown)
    %c1 = arith.constant 1 : index loc(unknown)
    %cst_1 = arith.constant dense<0.12753105163574219> : tensor<1xf64> loc(unknown)
    %c0 = arith.constant 0 : index loc(unknown)
    %1 = deepgengraph.mask starts = [%arg3, %c0], sizes = [32, 4096], type = f64 {
    ^bb0(%arg7: index loc(unknown), %arg8: index loc(unknown)):
      %9 = arith.addi %arg7, %c1 : index loc(unknown)
      %10 = arith.cmpi ule, %9, %arg8 : index loc(unknown)
      %11 = scf.if %10 -> (f64) {
        scf.yield %cst_0 : f64 loc(unknown)
      } else {
        scf.yield %cst : f64 loc(unknown)
      } loc(unknown)
      deepgengraph.mask_yield %11 : f64 loc(unknown)
    } : (index, index) -> tensor<32x4096xf64> loc(unknown)
    %2 = deepgengraph.permute %arg5, dims = [0, 1] : (tensor<32x128xf64>) -> tensor<32x128xf64> loc(unknown)
    %3 = deepgengraph.permute %arg6, dims = [1, 0] : (tensor<4096x128xf64>) -> tensor<128x4096xf64> loc(unknown)
    %4 = deepgengraph.mul %2, %cst_1 : (tensor<32x128xf64>, tensor<1xf64>) -> tensor<32x128xf64> loc(unknown)
    %5 = deepgengraph.dot %4, %3 : (tensor<32x128xf64>, tensor<128x4096xf64>) -> tensor<32x4096xf64> loc(unknown)
    %6 = deepgengraph.add %5, %1 : (tensor<32x4096xf64>, tensor<32x4096xf64>) -> tensor<32x4096xf64> loc(unknown)
    %7 = deepgengraph.exp2 %6 : (tensor<32x4096xf64>) -> tensor<32x4096xf64> loc(unknown)
    %8 = deepgengraph.reduce(%7), dim = 1, op =  ADD, keep_dim = true : (tensor<32x4096xf64>) -> tensor<32x1xf64> loc(unknown)
    deepgengraph.parallel_yield %8 : tensor<32x1xf64> loc(unknown)
  } : (tensor<1x4096x32x128xf64>, tensor<1x4096x32x128xf64>) -> tensor<1x32x4096x1xf64> loc(unknown)
  deepgengraph.return %0 : tensor<1x32x4096x1xf64> loc(unknown)
} loc(unknown)

// -------- after add_deepgengraph_tiling
 deepgengraph.kernel @Attn_p1(%arg0: tensor<1x4096x32x128xf64> {deepgengraph.erased_type = f16} loc(unknown), %arg1: tensor<1x4096x32x128xf64> {deepgengraph.erased_type = f16} loc(unknown)) -> (tensor<1x32x4096x1xf64> {deepgengraph.erased_type = f32}) attributes {parallel_map = [{arg_dims = [0, 0], res_dims = [0], size_per_unit = 1 : i64, unit_num = 1 : i64}, {arg_dims = [1, -1], res_dims = [2], size_per_unit = 32 : i64, unit_num = 128 : i64}, {arg_dims = [2, 2], res_dims = [1], size_per_unit = 1 : i64, unit_num = 32 : i64}]} {
  %0 = deepgengraph.parallel_for %arg0, %arg1 {
  ^bb0(%arg2: index loc(unknown), %arg3: index loc(unknown), %arg4: index loc(unknown), %arg5: tensor<32x128xf64> loc(unknown), %arg6: tensor<4096x128xf64> loc(unknown)):
    %cst = arith.constant 0.000000e+00 : f64 loc(unknown)
    %cst_0 = arith.constant 0xFFF0000000000000 : f64 loc(unknown)
    %c1 = arith.constant 1 : index loc(unknown)
    %cst_1 = arith.constant dense<0.12753105163574219> : tensor<1xf64> loc(unknown)
    %1 = deepgengraph.permute %arg6, dims = [1, 0] : (tensor<4096x128xf64>) -> tensor<128x4096xf64> loc(unknown)
    %2 = deepgengraph.mul %arg5, %cst_1 : (tensor<32x128xf64>, tensor<1xf64>) -> tensor<32x128xf64> loc(unknown)
    %3 = deepgengraph.zero shape = [32, 1], type = f64 : () -> tensor<32x1xf64> loc(unknown)
    %4 = deepgengraph.block_for lb = 0, ub = 4096, step = 128, args = [%1], dims = [1], init = [%3] {
    ^bb0(%arg7: index loc(unknown), %arg8: tensor<128x128xf64> loc(unknown), %arg9: tensor<32x1xf64> loc(unknown)):
      %5 = deepgengraph.mask starts = [%arg3, %arg7], sizes = [32, 128], type = f64 {
      ^bb0(%arg10: index loc(unknown), %arg11: index loc(unknown)):
        %10 = arith.addi %arg10, %c1 : index loc(unknown)
        %11 = arith.cmpi ule, %10, %arg11 : index loc(unknown)
        %12 = scf.if %11 -> (f64) {
          scf.yield %cst_0 : f64 loc(unknown)
        } else {
          scf.yield %cst : f64 loc(unknown)
        } loc(unknown)
        deepgengraph.mask_yield %12 : f64 loc(unknown)
      } : (index, index) -> tensor<32x128xf64> loc(unknown)
      %6 = deepgengraph.dot %2, %arg8 : (tensor<32x128xf64>, tensor<128x128xf64>) -> tensor<32x128xf64> loc(unknown)
      %7 = deepgengraph.add %6, %5 : (tensor<32x128xf64>, tensor<32x128xf64>) -> tensor<32x128xf64> loc(unknown)
      %8 = deepgengraph.exp2 %7 : (tensor<32x128xf64>) -> tensor<32x128xf64> loc(unknown)
      %9 = deepgengraph.reduce(%8, init = %arg9), dim = 1, op =  ADD, keep_dim = true : (tensor<32x128xf64>, tensor<32x1xf64>) -> tensor<32x1xf64> loc(unknown)
      deepgengraph.block_yield block_outs = [], iter_outs = [%9] : tensor<32x1xf64> loc(unknown)
    } : (tensor<128x4096xf64>, tensor<32x1xf64>) -> tensor<32x1xf64> loc(unknown)
    deepgengraph.parallel_yield %4 : tensor<32x1xf64> loc(unknown)
  } : (tensor<1x4096x32x128xf64>, tensor<1x4096x32x128xf64>) -> tensor<1x32x4096x1xf64> loc(unknown)
  deepgengraph.return %0 : tensor<1x32x4096x1xf64> loc(unknown)
} loc(unknown)

// -------- after add_deepgengraph_dynamic_for
 deepgengraph.kernel @Attn_p3(%arg0: tensor<1x4096x32x128xf64> {deepgengraph.erased_type = f16} loc(unknown), %arg1: tensor<1x4096x32x128xf64> {deepgengraph.erased_type = f16} loc(unknown), %arg2: tensor<1x4096x32x128xf64> {deepgengraph.erased_type = f16} loc(unknown)) -> (tensor<1x32x4096x1xf64> {deepgengraph.erased_type = f32}, tensor<1x4096x32x128xf64> {deepgengraph.erased_type = f16}) attributes {parallel_map = [{arg_dims = [0, 0, 0], res_dims = [0, 0], size_per_unit = 1 : i64, unit_num = 1 : i64}, {arg_dims = [1, -1, -1], res_dims = [2, 1], size_per_unit = 32 : i64, unit_num = 128 : i64}, {arg_dims = [2, 2, 2], res_dims = [1, 2], size_per_unit = 1 : i64, unit_num = 32 : i64}]} {
  %0:2 = deepgengraph.parallel_for %arg0, %arg1, %arg2 {
  ^bb0(%arg3: index loc(unknown), %arg4: index loc(unknown), %arg5: index loc(unknown), %arg6: tensor<32x128xf64> loc(unknown), %arg7: tensor<4096x128xf64> loc(unknown), %arg8: tensor<4096x128xf64> loc(unknown)):
    %c4096 = arith.constant 4096 : index loc(unknown)
    %c0 = arith.constant 0 : index loc(unknown)
    %cst = arith.constant 0.000000e+00 : f64 loc(unknown)
    %cst_0 = arith.constant 0xFFF0000000000000 : f64 loc(unknown)
    %c1 = arith.constant 1 : index loc(unknown)
    %cst_1 = arith.constant dense<0.12753105163574219> : tensor<1xf64> loc(unknown)
    %1 = deepgengraph.permute %arg8, dims = [1, 0] : (tensor<4096x128xf64>) -> tensor<128x4096xf64> loc(unknown)
    %2 = deepgengraph.mul %arg6, %cst_1 : (tensor<32x128xf64>, tensor<1xf64>) -> tensor<32x128xf64> loc(unknown)
    %3 = deepgengraph.zero shape = [32, 128], type = f64 : () -> tensor<32x128xf64> loc(unknown)
    %4 = deepgengraph.zero shape = [32, 1], type = f64 : () -> tensor<32x1xf64> loc(unknown)
    %5:2 = deepgengraph.dynamic_block_for lb = %c0, ub = %c4096, step = 128, args = [%1, %arg7], dims = [1, 0], init = [%3, %4] {
    ^bb0(%arg9: index loc(unknown), %arg10: tensor<128x128xf64> loc(unknown), %arg11: tensor<128x128xf64> loc(unknown), %arg12: tensor<32x128xf64> loc(unknown), %arg13: tensor<32x1xf64> loc(unknown)):
      %7 = deepgengraph.dot %2, %arg10 : (tensor<32x128xf64>, tensor<128x128xf64>) -> tensor<32x128xf64> loc(unknown)
      %8 = deepgengraph.mask starts = [%arg4, %arg9], sizes = [32, 128], type = f64 {
      ^bb0(%arg14: index loc(unknown), %arg15: index loc(unknown)):
        %14 = arith.addi %arg14, %c1 : index loc(unknown)
        %15 = arith.cmpi ule, %14, %arg15 : index loc(unknown)
        %16 = arith.select %15, %cst_0, %cst : f64 loc(unknown)
        deepgengraph.mask_yield %16 : f64 loc(unknown)
      } : (index, index) -> tensor<32x128xf64> loc(unknown)
      %9 = deepgengraph.add %7, %8 : (tensor<32x128xf64>, tensor<32x128xf64>) -> tensor<32x128xf64> loc(unknown)
      %10 = deepgengraph.exp2 %9 : (tensor<32x128xf64>) -> tensor<32x128xf64> loc(unknown)
      %11 = deepgengraph.reduce(%10, init = %arg13), dim = 1, op =  ADD, keep_dim = true : (tensor<32x128xf64>, tensor<32x1xf64>) -> tensor<32x1xf64> loc(unknown)
      %12 = deepgengraph.dot %10, %arg11 : (tensor<32x128xf64>, tensor<128x128xf64>) -> tensor<32x128xf64> loc(unknown)
      %13 = deepgengraph.add %arg12, %12 : (tensor<32x128xf64>, tensor<32x128xf64>) -> tensor<32x128xf64> loc(unknown)
      deepgengraph.block_yield block_outs = [], iter_outs = [%13, %11] : tensor<32x128xf64>, tensor<32x1xf64> loc(unknown)
    } : (index, index, tensor<128x4096xf64>, tensor<4096x128xf64>, tensor<32x128xf64>, tensor<32x1xf64>) -> (tensor<32x128xf64>, tensor<32x1xf64>) loc(unknown)
    %6 = deepgengraph.div %5#0, %5#1 : (tensor<32x128xf64>, tensor<32x1xf64>) -> tensor<32x128xf64> loc(unknown)
    deepgengraph.parallel_yield %5#1, %6 : tensor<32x1xf64>, tensor<32x128xf64> loc(unknown)
  } : (tensor<1x4096x32x128xf64>, tensor<1x4096x32x128xf64>, tensor<1x4096x32x128xf64>) -> (tensor<1x32x4096x1xf64>, tensor<1x4096x32x128xf64>) loc(unknown)
  deepgengraph.return %0#0, %0#1 : tensor<1x32x4096x1xf64>, tensor<1x4096x32x128xf64> loc(unknown)
} loc(unknown)

