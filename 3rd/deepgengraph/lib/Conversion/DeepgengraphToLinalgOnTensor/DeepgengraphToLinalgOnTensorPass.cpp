#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

#include "deepgengraph/Dialect/Deepgengraph/IR/DeepgengraphDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "dbg.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdint>
#include <string>
#include <vector>
#include <algorithm>
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

namespace mlir::deepgengraph {

#define GEN_PASS_DEF_CONVERTDEEPGENGRAPHTOLINALGONTENSOR
#include "deepgengraph/Conversion/DeepgengraphToLinalgOnTensor/Passes.h.inc"

// ================ OpConversionPatterns

// 三角矩阵转换为 linalg.generic
struct TriluOpConversion : public OpConversionPattern<deepgengraph::TriluOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(TriluOp triluOp, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = triluOp->getLoc();

    // 1. 收集信息
    auto outType = triluOp->getResult(0).getType();
    auto tensorType = mlir::dyn_cast<TensorType>(outType);
    Type dtype = tensorType.getElementType();
    
    // 获取 shape 和属性
    ArrayRef<int64_t> shape = triluOp.getShape();
    int64_t diagonal = triluOp.getDiagonal();
    bool isUpper = triluOp.getIsUpper();

    // 2. 创建输出张量 (Init Tensor)
    // 使用 tensor.empty 创建一个未初始化的张量作为 linalg 的输出容器
    Value zeroTensor = rewriter.create<tensor::EmptyOp>(loc, shape, dtype);

    // 3. 定义 linalg.generic 配置
    // 2D Identity Map (d0, d1) -> (d0, d1)
    AffineMap idMap = AffineMap::getMultiDimIdentityMap(2, rewriter.getContext());
    SmallVector<AffineMap> indexingMaps = {idMap}; 
    SmallVector<utils::IteratorType> iteratorTypes(2, utils::IteratorType::parallel);

    // 4. 创建 linalg.generic 操作主体
    // Inputs: {} (无)
    // Outputs: {zeroTensor}
    mlir::linalg::GenericOp genericOp = rewriter.create<mlir::linalg::GenericOp>(
        loc, 
        outType,                  // Result Types
        ValueRange{},             // Inputs
        ValueRange{zeroTensor},   // Outputs
        indexingMaps, 
        iteratorTypes
    );

    // 5. 构建 Region Body
    Region &region = genericOp.getRegion();
    // 关键修正：Block 参数必须匹配 Inputs/Outputs 的元素类型
    // 这里只有 1 个 Output，所以 Block 只有一个参数 (arg0: 对应 output 当前位置的元素，通常未初始化)
    Block *block = rewriter.createBlock(&region, region.end(), {dtype}, {loc});
    
    rewriter.setInsertionPointToStart(block);

    // 6. 在 Body 内部获取索引 (代替原来的 Block 参数)
    Value i = rewriter.create<linalg::IndexOp>(loc, 0); // Dim 0 (行)
    Value j = rewriter.create<linalg::IndexOp>(loc, 1); // Dim 1 (列)

    // 7. 准备常量
    auto triluValAttr = triluOp.getVal();
    Value triluConst = rewriter.create<arith::ConstantOp>(loc, triluValAttr);
    
    auto zeroAttr = rewriter.getZeroAttr(dtype);
    Value zeroConst = rewriter.create<arith::ConstantOp>(loc, zeroAttr);

    // 8. 计算条件逻辑: i vs j + diagonal
    Value diagConst = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(diagonal));
    Value jShifted = rewriter.create<arith::AddIOp>(loc, j, diagConst);

    arith::CmpIPredicate pred;
    if (isUpper) {
        // Upper: 保留 i <= j + k
        pred = arith::CmpIPredicate::sle;
    } else {
        // Lower: 保留 i >= j + k
        pred = arith::CmpIPredicate::sge;
    }

    Value cond = rewriter.create<arith::CmpIOp>(loc, pred, i, jShifted);

    // 9. 选择值 (Select)
    // 使用 arith.select 代替 scf.if，更高效且代码更少
    Value result = rewriter.create<arith::SelectOp>(loc, cond, triluConst, zeroConst);

    // 10. Yield 结果
    rewriter.create<linalg::YieldOp>(loc, result);

    // 11. 替换原操作
    // replaceOp 会自动处理 replaceAllUsesWith 和 eraseOp 的逻辑
    rewriter.replaceOp(triluOp, genericOp);

    return success();
  }
};

// PermuteOp 转换为 linalg.transpose (or memref.transpose?)
struct PermuteOpConversion : public OpConversionPattern<deepgengraph::PermuteOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(PermuteOp permuteOp, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = permuteOp->getLoc();
      // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value input, Value init, DenseI64ArrayAttr permutation, ArrayRef<NamedAttribute> attributes = {});
    
    auto retTensor = permuteOp->getResult(0);
    auto retTensorType = mlir::dyn_cast<TensorType>(retTensor.getType());
    auto out = rewriter.create<tensor::EmptyOp>(loc,retTensorType.getShape(),retTensorType.getElementType());

    auto newop = rewriter.create<linalg::TransposeOp>(loc, 
      permuteOp->getOperand(0),
      out->getResult(0),
      permuteOp.getDimsAttr()
      );
    rewriter.replaceOp(permuteOp, newop);
    return success();
  }
};


// DotOp 转换为 linalg.matmul(error：matmul仅支持 2D。不通用)
struct DivOpConversion : public OpConversionPattern<deepgengraph::DivOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(DivOp divOp, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = divOp->getLoc();
    // ... (日志代码) ...
    auto res = divOp->getResult(0);
    
    // 获取 Shape
    auto lhs_shape = mlir::cast<mlir::TensorType>(divOp.getLhs().getType()).getShape();
    auto rhs_shape = mlir::cast<mlir::TensorType>(divOp.getRhs().getType()).getShape();
    auto resShape = mlir::cast<TensorType>(res.getType()).getShape();

    // 生成 Maps
    auto affineMaps = getLeftAlignedAffineMap(rewriter, lhs_shape, rhs_shape, resShape);

    auto kind = linalg::ElementwiseKindAttr::get(rewriter.getContext(), ::mlir::linalg::ElementwiseKind::div);
    
    // 创建 Generic Op / Elementwise Op
    // 注意：linalg::ElementwiseOp 在某些 LLVM 版本可能是指 linalg.generic 的一种特化 helper
    auto dtype = mlir::dyn_cast<mlir::TensorType>(divOp->getResult(0).getType()).getElementType();
    auto newRes = rewriter.create<tensor::EmptyOp>(loc, resShape, dtype);
    auto newOp = rewriter.create<linalg::ElementwiseOp>(
        loc,
        divOp->getOperands(),
        newRes->getResult(0),
        kind,
        rewriter.getAffineMapArrayAttr(affineMaps)
    );
    
    rewriter.replaceOp(divOp, newOp);
    return success();
  }

  std::vector<mlir::AffineMap> getLeftAlignedAffineMap(
    mlir::ConversionPatternRewriter& rewriter,
    llvm::ArrayRef<int64_t> lhsShape, 
    llvm::ArrayRef<int64_t> rhsShape, 
    llvm::ArrayRef<int64_t> retShape 
  ) const 
  {
    // 1. 确定循环空间的维度 (Loop Rank)，通常由输出张量决定
    unsigned loopRank = retShape.size(); 
    
    // 创建基础的维度表达式 (d0, d1, d2, d3...)
    std::vector<mlir::AffineExpr> dims;
    for(unsigned i = 0; i < loopRank; ++i){
      dims.push_back(rewriter.getAffineDimExpr(i));
    }

    // 2. 构建 LHS Map (通常是 Identity，如果 LHS 形状与 Res 一致)
    // Map: (d0, d1, d2, d3) -> (d0, d1, d2, d3)
    auto lhsMap = mlir::AffineMap::getMultiDimIdentityMap(loopRank, rewriter.getContext());

    // 3. 构建 RHS Map (处理 Broadcast)
    // 这里的关键是：Map 的输入必须是 loopRank (4)，输出是 rhsShape 的秩 (1)
    std::vector<mlir::AffineExpr> rhsExprs;
    
    // 简单的 Broadcast 匹配逻辑 (Prefix Match，沿用你的逻辑)
    // 注意：更健壮的 broadcasting 通常是 Right-Aligned (NumPy style)，
    // 但根据你的用例 (deepgengraph)，如果你的语义是左对齐匹配：
    for (size_t i = 0; i < rhsShape.size(); ++i) {
        if (i < lhsShape.size() && lhsShape[i] == rhsShape[i]) {
            // 维度匹配，使用对应的循环变量 d_i
            rhsExprs.push_back(dims[i]);
        } else if (rhsShape[i] == 1) {
            // 如果 RHS 是 1，通常在 AffineMap 中仍映射到该维度，但由 Runtime 处理 stride=0
            // 或者如果这个 1 是要被广播的，有时也映射到 dims[i]
            rhsExprs.push_back(rewriter.getAffineConstantExpr(0));
        } else {
            // 维度不匹配且不为1，这是非法的 Broadcast，实际代码中可能需要 assert
            // 这里为了安全，暂且 push 对应的 dim
            rhsExprs.push_back(dims[i]);
        }
    }

    // [关键修改]
    // 第一个参数必须是 loopRank (4)，而不是 rhsExprs.size() (1)
    // 结果 Map 类似于: (d0, d1, d2, d3) -> (d0)
    auto rhsMap = mlir::AffineMap::get(loopRank, 0, rhsExprs, rewriter.getContext());

    // 4. 构建 Res Map
    auto resMap = mlir::AffineMap::getMultiDimIdentityMap(loopRank, rewriter.getContext());

    std::vector<mlir::AffineMap> ret = {lhsMap, rhsMap, resMap};
    return ret;
  }

};


struct ExpOpConversion : public OpConversionPattern<deepgengraph::ExpOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(deepgengraph::ExpOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
      // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
    auto retType =  mlir::cast<mlir::TensorType>(op->getResult(0).getType());
    auto retShape = retType.getShape();
    auto dtype = retType.getElementType();
    auto emptyOp = rewriter.create<tensor::EmptyOp>(loc, retShape, dtype);

    std::vector<AffineExpr> dims ;
    for(int i=0;i<retShape.size();++i){
      dims.push_back(rewriter.getAffineDimExpr(i));
    }
    auto inMap = AffineMap::get(retShape.size(),0,dims,rewriter.getContext());
    auto outMap = mlir::AffineMap::getMultiDimIdentityMap(retShape.size(), rewriter.getContext());
    auto kind = linalg::ElementwiseKindAttr::get(rewriter.getContext(), ::mlir::linalg::ElementwiseKind::exp);
    std::vector<AffineMap> map = {inMap, outMap};
    auto newOp = rewriter.create<linalg::ElementwiseOp>(
        loc,
        op->getOperands(),
        emptyOp->getResult(0),
        kind,
        rewriter.getAffineMapArrayAttr(map)
    );
    rewriter.replaceOp(op, newOp);
    return success();
  }

};


struct AddOpConversion : public OpConversionPattern<deepgengraph::AddOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(deepgengraph::AddOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
      // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
    auto retType =  mlir::cast<mlir::TensorType>(op->getResult(0).getType());
    auto retShape = retType.getShape();
    auto dtype = retType.getElementType();
    auto emptyOp = rewriter.create<tensor::EmptyOp>(loc, retShape, dtype);

    std::vector<AffineExpr> retDims ;
    for(int i=0;i<retShape.size();++i){
      retDims.push_back(rewriter.getAffineDimExpr(i));
    }
    std::vector<int64_t> lhsShape = mlir::cast<TensorType>(op.getLhs().getType()).getShape();
    std::vector<int64_t> rhsShape = mlir::cast<TensorType>(op.getRhs().getType()).getShape();
    bool isLhsNeedBroadcast = false;
    int diff = lhsShape.size() - rhsShape.size();
    
    std::vector<AffineExpr> lhsDims, rhsDims ;
    // prefill
    int i0 = lhsShape.size() - 1;
    int i1 = rhsShape.size() - 1;
    int i = retShape.size() - 1;
    while(i >= 0 || i1 >= 0 || i0 >= 0){
      if(i0 >= 0){
        if(retShape[i] == lhsShape[i0]){
          lhsDims.push_back(rewriter.getAffineDimExpr(i));
        }
        else{
          lhsDims.push_back(rewriter.getAffineConstantExpr(0));
        }
      }
      else{
        // lhsDims.push_back(rewriter.getAffineConstantExpr(0));
      }
      if(i1 >= 0){
        if(retShape[i] == rhsShape[i1]){
          rhsDims.push_back(rewriter.getAffineDimExpr(i));
        }
        else{
          rhsDims.push_back(rewriter.getAffineConstantExpr(0));
        }
      }
      else{
        // rhsDims.push_back(rewriter.getAffineConstantExpr(0));
      }
      --i;--i0;--i1;
    }

    std::reverse(lhsDims.begin(),lhsDims.end());
    std::reverse(rhsDims.begin(),rhsDims.end());

    AffineMap lhsMap = AffineMap::get(retShape.size(), 0, lhsDims, rewriter.getContext());
    AffineMap rhsMap = AffineMap::get(retShape.size(), 0, rhsDims, rewriter.getContext());
    AffineMap outMap = mlir::AffineMap::getMultiDimIdentityMap(retShape.size(), rewriter.getContext());
    auto kind = linalg::ElementwiseKindAttr::get(rewriter.getContext(), ::mlir::linalg::ElementwiseKind::add);
    std::vector<AffineMap> map = {lhsMap, rhsMap, outMap};
    auto newOp = rewriter.create<linalg::ElementwiseOp>(
        loc,
        op->getOperands(),
        emptyOp->getResult(0),
        kind,
        rewriter.getAffineMapArrayAttr(map)
    );
    rewriter.replaceOp(op, newOp);
    return success();
  }

};


struct DotOpConversion : public OpConversionPattern<deepgengraph::DotOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(deepgengraph::DotOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    
    // 获取操作数和类型
    Value lhs = adaptor.getLhs(); // 注意使用 adaptor 获取转换后的操作数
    Value rhs = adaptor.getRhs();
    auto lhsType = mlir::cast<RankedTensorType>(lhs.getType());
    auto rhsType = mlir::cast<RankedTensorType>(rhs.getType());
    auto outputType = mlir::cast<RankedTensorType>(op.getResult().getType());

    int64_t rank = lhsType.getRank();
    int64_t batchDimsCount = rank - 2;

    // Batch MatMul 需要的循环总数 = Batch数 + M + N + K
    // 对于 Rank=4 的输入 (B, H, M, K)，我们需要 5 层循环
    int64_t nLoops = batchDimsCount + 3; 

    // 1. 构建 AffineMap
    // 循环索引定义： [BatchDims..., M, N, K]
    // 索引映射：
    // Batch: 0 到 batchDimsCount-1
    // M:     batchDimsCount
    // N:     batchDimsCount + 1
    // K:     batchDimsCount + 2
    
    SmallVector<AffineExpr> batchExprs;
    for (int i = 0; i < batchDimsCount; ++i) {
      batchExprs.push_back(rewriter.getAffineDimExpr(i));
    }
    AffineExpr mExpr = rewriter.getAffineDimExpr(batchDimsCount);
    AffineExpr nExpr = rewriter.getAffineDimExpr(batchDimsCount + 1);
    AffineExpr kExpr = rewriter.getAffineDimExpr(batchDimsCount + 2);

    // LHS Map: (Batch..., M, K)
    SmallVector<AffineExpr> lhsExprs = batchExprs;
    lhsExprs.push_back(mExpr);
    lhsExprs.push_back(kExpr);

    // RHS Map: (Batch..., K, N)
    SmallVector<AffineExpr> rhsExprs = batchExprs;
    rhsExprs.push_back(kExpr);
    rhsExprs.push_back(nExpr);

    // Out Map: (Batch..., M, N)
    SmallVector<AffineExpr> outExprs = batchExprs;
    outExprs.push_back(mExpr);
    outExprs.push_back(nExpr);

    auto lhsMap = AffineMap::get(nLoops, 0, lhsExprs, rewriter.getContext());
    auto rhsMap = AffineMap::get(nLoops, 0, rhsExprs, rewriter.getContext());
    auto outMap = AffineMap::get(nLoops, 0, outExprs, rewriter.getContext());

    SmallVector<AffineMap> indexingMaps = {lhsMap, rhsMap, outMap};

    // 2. 构建 Iterator Types
    // Batch, M, N 是 Parallel; K 是 Reduction
    SmallVector<utils::IteratorType> iteratorTypes(nLoops, utils::IteratorType::parallel);
    iteratorTypes.back() = utils::IteratorType::reduction; // 最后一个维度 K 是 reduction

    // 3. 初始化输出 Tensor (Accumulator)
    // linalg.generic 的输出如果作为累加器，必须初始化为 0。
    // 使用 tensor.empty 创建形状，然后用 linalg.fill 填充 0。
    Value emptyTensor = rewriter.create<tensor::EmptyOp>(
        loc, outputType.getShape(), outputType.getElementType());

    // 创建零值常量
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(outputType.getElementType()));

    // 填充 0
    Value zeroedTensor = rewriter.create<linalg::FillOp>(
        loc, zero, emptyTensor).getResult(0);

    // 4. 创建 Generic Op
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc,
        outputType,             // Result Types
        ValueRange{lhs, rhs},   // Inputs
        ValueRange{zeroedTensor}, // Outputs (Initialized Accumulator)
        indexingMaps,
        iteratorTypes,
        [&](OpBuilder &builder, Location innerLoc, ValueRange args) {
          // args[0]: lhs element
          // args[1]: rhs element
          // args[2]: accumulator (current output value)
          Value mul = builder.create<arith::MulFOp>(innerLoc, args[0], args[1]);
          Value add = builder.create<arith::AddFOp>(innerLoc, args[2], mul);
          builder.create<linalg::YieldOp>(innerLoc, add);
        }
    );

    // 替换原 Op
    rewriter.replaceOp(op, genericOp);
    return success();
  }
};


struct ConvertOpConversion : public OpConversionPattern<deepgengraph::ConvertOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(deepgengraph::ConvertOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    mlir::Type dstType = op.getDstTypeAttr().getValue();
    auto srcTensor = op->getOperand(0);
    auto dstShape = mlir::cast<TensorType>(op->getResult(0).getType()).getShape();
    
    auto newDst = rewriter.create<tensor::EmptyOp>(loc, dstShape, dstType);
    // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
    std::vector<Value> ins = {srcTensor};
    std::vector<Value> outs = {newDst};
    auto newOp = rewriter.create<linalg::CopyOp>(loc, ins, outs);
    rewriter.replaceOp(op, newOp);
    return success();
  }

};


struct KernelOpConversionPattern : public OpConversionPattern<deepgengraph::KernelOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(deepgengraph::KernelOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    std::vector<Type> argTypes = op.getArgumentTypes();
    std::vector<Type> resTypes = op.getResultTypes();
    // for(auto r : resTypes){
    //   argTypes.push_back(r);
    // }
    // std::vector<Type> resTypesEmpty;
    auto funcType = rewriter.getFunctionType(argTypes, resTypes);
    auto name = op.getSymName();
    auto funcOp = rewriter.create<mlir::func::FuncOp>(op->getLoc(), name, funcType);
    funcOp.setVisibility(SymbolTable::Visibility::Public);
    // 3. 关键步骤：移植 Region (函数体)
    // 我们将 kernel 的 Region 直接移动到 funcOp 中，避免重新创建 Block 和参数
    Region &kernelRegion = op.getBody();
    Region &funcRegion = funcOp.getBody();
    
    // 将 kernelRegion 的所有 Block 移动到 funcRegion 的末尾
    rewriter.inlineRegionBefore(kernelRegion, funcRegion, funcRegion.end());

    // 4. 删除旧的 kernel op
    rewriter.eraseOp(op);

    return success();
  }

};

// Pattern 2: 将 deepgengraph.return 转换为 func.return
struct ReturnToFuncReturnPattern : public OpConversionPattern<deepgengraph::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(deepgengraph::ReturnOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // 创建 func.return，使用 adaptor 获取已经重映射的操作数
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};



struct BroadcastableBinaryOpInterfaceRewritePattern : public OpInterfaceConversionPattern<deepgengraph::BroadcastableBinaryOpInterface> {
  using OpInterfaceConversionPattern::OpInterfaceConversionPattern;

  LogicalResult matchAndRewrite(
    BroadcastableBinaryOpInterface op,
    ArrayRef<Value> operands,
    ConversionPatternRewriter& rewriter) const override 
  {
    auto loc = op->getLoc();
    llvm::outs() << "------------Interface : enter " << op << "\n"; llvm::outs().flush();
    auto lhsBShape = op.getLhsBroadcastedShape();
    auto rhsBShape = op.getRhsBroadcastedShape();
    llvm::outs() << "lhsBShape = ";
    for(auto e : lhsBShape){
      llvm::outs() << e << ",";
    }
    llvm::outs() << "\n";llvm::outs().flush();
    llvm::outs() << "rhsBShape = ";
    for(auto e : rhsBShape){
      llvm::outs() << e << ",";
    }
    llvm::outs() << "\n";llvm::outs().flush();
    auto retType =  mlir::cast<mlir::TensorType>(op->getResult(0).getType());
    auto retShape = retType.getShape();
    auto dtype = retType.getElementType();
    auto emptyOp = rewriter.create<tensor::EmptyOp>(loc, retShape, dtype);

    std::vector<AffineExpr> retDims ;
    for(int i=0;i<retShape.size();++i){
      retDims.push_back(rewriter.getAffineDimExpr(i));
    }
    std::vector<int64_t> lhsShape = mlir::cast<TensorType>(op.getLhs().getType()).getShape();
    std::vector<int64_t> rhsShape = mlir::cast<TensorType>(op.getRhs().getType()).getShape();
    bool isLhsNeedBroadcast = false;
    int diff = lhsShape.size() - rhsShape.size();
    
    std::vector<AffineExpr> lhsDims, rhsDims ;
    // prefill
    int i0 = lhsShape.size() - 1;
    int i1 = rhsShape.size() - 1;
    int i = retShape.size() - 1;
    while(i >= 0 || i1 >= 0 || i0 >= 0){
      if(i0 >= 0){
        if(retShape[i] == lhsShape[i0]){
          lhsDims.push_back(rewriter.getAffineDimExpr(i));
        }
        else{
          lhsDims.push_back(rewriter.getAffineConstantExpr(0));
        }
      }
      else{
        // lhsDims.push_back(rewriter.getAffineConstantExpr(0));
      }
      if(i1 >= 0){
        if(retShape[i] == rhsShape[i1]){
          rhsDims.push_back(rewriter.getAffineDimExpr(i));
        }
        else{
          rhsDims.push_back(rewriter.getAffineConstantExpr(0));
        }
      }
      else{
        // rhsDims.push_back(rewriter.getAffineConstantExpr(0));
      }
      --i;--i0;--i1;
    }

    std::reverse(lhsDims.begin(),lhsDims.end());
    std::reverse(rhsDims.begin(),rhsDims.end());

    AffineMap lhsMap = AffineMap::get(retShape.size(), 0, lhsDims, rewriter.getContext());
    AffineMap rhsMap = AffineMap::get(retShape.size(), 0, rhsDims, rewriter.getContext());
    AffineMap outMap = mlir::AffineMap::getMultiDimIdentityMap(retShape.size(), rewriter.getContext());

    mlir::linalg::ElementwiseKind enumKind;
    if(mlir::isa<deepgengraph::AddOp>(op)){
      enumKind = mlir::linalg::ElementwiseKind::add;
    }
    else if(mlir::isa<deepgengraph::SubOp>(op) ){
      enumKind = mlir::linalg::ElementwiseKind::sub;
    }
    else if(mlir::isa<deepgengraph::MulOp>(op) ){
      enumKind = mlir::linalg::ElementwiseKind::mul;
    }
    else if(mlir::isa<deepgengraph::DivOp>(op) ){
      enumKind = mlir::linalg::ElementwiseKind::div;
    }
    else if(mlir::isa<deepgengraph::PowOp>(op) ){
      enumKind = mlir::linalg::ElementwiseKind::powf;
    }
    else{
      return emitError(loc, "Unsupported op!");
    }

    auto kind = linalg::ElementwiseKindAttr::get(rewriter.getContext(), enumKind);
    std::vector<AffineMap> map = {lhsMap, rhsMap, outMap};
    auto newOp = rewriter.create<linalg::ElementwiseOp>(
        loc,
        op->getOperands(),
        emptyOp->getResult(0),
        kind,
        rewriter.getAffineMapArrayAttr(map)
    );
    rewriter.replaceOp(op, newOp);
    return success();
  }

};



struct ElementwiseUnaryOpConversion : public OpInterfaceConversionPattern<deepgengraph::ElementwiseUinaryOpInterface>  {
  using OpInterfaceConversionPattern::OpInterfaceConversionPattern;

  LogicalResult matchAndRewrite(ElementwiseUinaryOpInterface op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
      // static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ValueRange inputs, ValueRange outputs, ArrayRef<NamedAttribute> attributes = {});
    auto retType =  mlir::cast<mlir::TensorType>(op->getResult(0).getType());
    auto retShape = retType.getShape();
    auto dtype = retType.getElementType();
    auto emptyOp = rewriter.create<tensor::EmptyOp>(loc, retShape, dtype);

    std::vector<AffineExpr> dims ;
    for(int i=0;i<retShape.size();++i){
      dims.push_back(rewriter.getAffineDimExpr(i));
    }
    auto inMap = AffineMap::get(retShape.size(),0,dims,rewriter.getContext());
    auto outMap = mlir::AffineMap::getMultiDimIdentityMap(retShape.size(), rewriter.getContext());
    mlir::linalg::ElementwiseKind enumKind;
    // TanhOp
    // ExpOp
    // Exp2Op
    // LogOp
    // Log2Op
    // NegOp
    // AbsOp
    if(mlir::isa<deepgengraph::ExpOp>(op)){
      enumKind = linalg::ElementwiseKind::exp;
    }
    else if(mlir::isa<deepgengraph::TanhOp>(op)){
      enumKind = linalg::ElementwiseKind::tanh;
    }
    else if(mlir::isa<deepgengraph::LogOp>(op)){
      enumKind = linalg::ElementwiseKind::log;
    }
    else if(mlir::isa<deepgengraph::NegOp>(op)){
      enumKind = linalg::ElementwiseKind::negf;
    }
    else if(mlir::isa<deepgengraph::AbsOp>(op)){
      enumKind = linalg::ElementwiseKind::abs;
    }
    else if(mlir::isa<deepgengraph::Log2Op>(op)){
      enumKind = linalg::ElementwiseKind::log;
    }
    else if(mlir::isa<deepgengraph::Exp2Op>(op)){
      enumKind = linalg::ElementwiseKind::exp;
    }
    else{
      return emitError(loc,"Unsupported Op");
    }
    auto kind = linalg::ElementwiseKindAttr::get(rewriter.getContext(), enumKind);
    std::vector<AffineMap> map = {inMap, outMap};
    
    auto newOp = rewriter.create<linalg::ElementwiseOp>(
        loc,
        op->getOperands(),
        emptyOp->getResult(0),
        kind,
        rewriter.getAffineMapArrayAttr(map)
    );
    rewriter.replaceOp(op, newOp);
    return success();
  }

};


class ConvertDeepgengraphToLinalgOnTensor : public impl::ConvertDeepgengraphToLinalgOnTensorBase<ConvertDeepgengraphToLinalgOnTensor>
{
public:
  void runOnOperation(){
    MLIRContext *context = &getContext();
    ModuleOp k = getOperation();
    ConversionTarget target(*context);

    // clang-format off
    target.addLegalDialect<
      deepgengraph::DeepgengraphDialect,
      tensor::TensorDialect,
      linalg::LinalgDialect,
      arith::ArithDialect,
      affine::AffineDialect,
      mlir::math::MathDialect,
      mlir::func::FuncDialect,
      scf::SCFDialect>();
    
    target.addIllegalDialect<DeepgengraphDialect>();
    RewritePatternSet patterns0(context);
    patterns0.add<
      BroadcastableBinaryOpInterfaceRewritePattern, ElementwiseUnaryOpConversion,
      TriluOpConversion, PermuteOpConversion,  DotOpConversion , ConvertOpConversion,
      KernelOpConversionPattern, ReturnToFuncReturnPattern
      >(context);
    if (failed(applyPartialConversion(k, target, std::move(patterns0)))){
      return signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createConvertDeepgengraphToLinalgOnTensorPass() {
  return std::make_unique<ConvertDeepgengraphToLinalgOnTensor>();
}



} // namespace mlir::deepgengraph