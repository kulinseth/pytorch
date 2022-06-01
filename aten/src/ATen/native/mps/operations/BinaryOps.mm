//  Copyright © 2022 Apple Inc.

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/Utils.h>
#include <ATen/AccumulateType.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>
#include <torch/library.h>
#include <c10/util/Optional.h>
#include <ATen/native/BinaryOps.h>

namespace at {
namespace native {
namespace mps {

struct BinaryOpCachedGraph : public MPSCachedGraph
{
  BinaryOpCachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
  MPSGraphTensor *primaryTensor = nil, *secondaryTensor = nil;
  MPSGraphTensor *alphaTensor = nil, *outputTensor = nil;
};

typedef MPSGraphTensor* (^BinaryOpBlock)(BinaryOpCachedGraph*, const MPSDataType, const MPSDataType, const MPSDataType);

// Assign MPSGraphTensor pointers the correct versions; cast if needed
void castInputsArithmeticOp(MPSGraph* mpsGraph,
                MPSGraphTensor* primaryTensor, MPSGraphTensor* secondaryTensor,
                MPSGraphTensor* &primary, MPSGraphTensor* &secondary,
                const MPSDataType self_dtype, const MPSDataType other_dtype,
                const MPSDataType acc_dtype) {

  if(self_dtype != acc_dtype)
    primary = [mpsGraph castTensor:primaryTensor toType:acc_dtype name:@"primary"];
  else
    primary = primaryTensor;
  if(other_dtype != acc_dtype)
    secondary = [mpsGraph castTensor:secondaryTensor toType:acc_dtype name:@"secondary"];
  else
    secondary = secondaryTensor;

}

void castInputsBooleanOp(MPSGraph* mpsGraph,
                         MPSGraphTensor* primaryTensor, MPSGraphTensor* secondaryTensor,
                         MPSGraphTensor* &primary, MPSGraphTensor* &secondary,
                         const MPSDataType self_dtype, const MPSDataType other_dtype) {

  if(self_dtype != other_dtype) {
    // Cast one to the type of other, so that they can be compared
    if(type_precedence_number(self_dtype) > type_precedence_number(other_dtype)) {
      primary = primaryTensor;
      secondary = [mpsGraph castTensor:secondaryTensor toType:self_dtype name:@"secondary"];
    }
    else {
      primary = [mpsGraph castTensor:primaryTensor toType:other_dtype name:@"primary"];
      secondary = secondaryTensor;
    }
  }
  else {
    primary = primaryTensor;
    secondary = secondaryTensor;
  }

}

// Assign a precedence to types for comparing in boolean operation
int type_precedence_number(const MPSDataType tensor_dtype) {

  switch (tensor_dtype) {
    case MPSDataTypeFloat32:
      return 7;
    case MPSDataTypeFloat16:
      return 6;
    case MPSDataTypeInt64:
      return 5;
    case MPSDataTypeInt32:
      return 4;
    case MPSDataTypeInt16:
      return 3;
    case MPSDataTypeInt8:
      return 2;
    case MPSDataTypeBool:
      return 1;
    default:
      TORCH_CHECK_TYPE(false, "Unsupported data type")
  }

}

// alpha is always 1.0 except when this function is called from add_sub_template()
// is_arithmetic lets us check for arithmetic ops, so we can cast the input to output type if needed
// because in arithmetic ops the input and output types match
void binaryOpTensor(const Tensor& self, const Tensor& other, const Scalar& alpha,
                    const Tensor& output, std::string op_name, bool is_arithmetic, BinaryOpBlock binaryBlock)
{
  // it's possible to receive empty tensors here
  if (self_t.numel() == 0 || other_t.numel() == 0) {
    return;
  }
  MPSStream* mpsStream = getCurrentMPSStream();

  const bool is_self_scalar = self_t.dim() == 0;
  const bool is_other_scalar = other_t.dim() == 0;

  Tensor self = is_self_scalar ? self_t : self_t.contiguous(at::MemoryFormat::Contiguous);
  Tensor other = is_other_scalar ? other_t : other_t.contiguous(at::MemoryFormat::Contiguous);

  // const MPSDataType self_dtype = getMPSScalarType((is_self_scalar && !is_other_scalar ? other_t : self_t).scalar_type());
  // const MPSDataType other_dtype = getMPSScalarType((!is_other_scalar ? other_t : self_t).scalar_type());

  const MPSDataType self_dtype = getMPSScalarType(self_t.scalar_type());
  const MPSDataType other_dtype = getMPSScalarType(other_t.scalar_type());
  const auto output_scalar_type = output.scalar_type();
  const MPSDataType output_dtype = getMPSScalarType(output_scalar_type);
  // Accumulate dtype
  const MPSDataType acc_dtype = getMPSScalarType(at::toAccumulateType(output_scalar_type, true));

  MPSGraphCache* cache_ = MPSGraphCache::getInstance();
  @autoreleasepool {
    string key = op_name + getTensorsStringKey({self, other}, /*use_scalar_value*/ false);
    BinaryOpCachedGraph* cachedGraph = static_cast<BinaryOpCachedGraph *>(cache_->LookUp(key));

    if(!cachedGraph) {
      MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ MPSCachedGraph* () {
        BinaryOpCachedGraph *newCachedGraph = nil;
        @autoreleasepool {
          MPSGraph* mpsGraph = make_mps_graph();
          newCachedGraph = new BinaryOpCachedGraph(mpsGraph);
          newCachedGraph->primaryTensor   = mpsGraphRankedPlaceHolder(mpsGraph, self_dtype , getMPSShape(self));
          newCachedGraph->secondaryTensor = mpsGraphRankedPlaceHolder(mpsGraph, other_dtype, getMPSShape(other));
          newCachedGraph->outputTensor = binaryBlock(newCachedGraph, self_dtype, other_dtype, acc_dtype);
          if(is_arithmetic && output_dtype != acc_dtype)
            newCachedGraph->outputTensor = [mpsGraph castTensor:newCachedGraph->outputTensor toType:output_dtype name:@"output"];
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<BinaryOpCachedGraph *>(tmpCachedGraph);
    }

    NSMutableDictionary *feeds = [[NSMutableDictionary new] autorelease];
    Placeholder selfPlaceholder;
    Placeholder otherPlaceholder;

    if (is_self_scalar) {
      feeds[cachedGraph->primaryTensor] = getMPSGraphTensorFromScalar(mpsStream, self.item(), self_dtype);
    } else {
      selfPlaceholder = Placeholder(cachedGraph->primaryTensor, self);
      feeds[selfPlaceholder.getMPSGraphTensor()] = selfPlaceholder.getMPSGraphTensorData();
    }
    if (is_other_scalar) {
      feeds[cachedGraph->secondaryTensor] = getMPSGraphTensorFromScalar(mpsStream, other.item(), other_dtype);
    } else {
      otherPlaceholder = Placeholder(cachedGraph->secondaryTensor, other);
      feeds[otherPlaceholder.getMPSGraphTensor()] = otherPlaceholder.getMPSGraphTensorData();
    }
    // 'cachedGraph->alphaTensor' is not nil only if add_sub_template() was called with an alpha value != 1.0
    if (cachedGraph->alphaTensor) {
      feeds[cachedGraph->alphaTensor] = getMPSGraphTensorFromScalar(mpsStream, alpha, other_dtype);
    }

    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor, output);
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };
    runMPSGraph(mpsStream, cachedGraph->graph(), feeds, results);
  }
}

void binaryOpScalar(const Tensor& self, const Scalar& other, const Scalar& alpha,
                    const Tensor& output, std::string op_name, bool is_arithmetic,  BinaryOpBlock binaryBlock)
{
  binaryOpTensor(self, wrapped_scalar_tensor(other), alpha, output, op_name, is_arithmetic, binaryBlock);
}

void div_mode_template(const Tensor& self, const Tensor& other,
                       c10::optional<c10::string_view> rounding_mode,
                       const Tensor& output, const string op_name)
{
  BinaryOpBlock div_mode_op_block = ^MPSGraphTensor* (BinaryOpCachedGraph* cachedGraph,
                                                      const MPSDataType self_dtype,
                                                      const MPSDataType other_dtype,
                                                      const MPSDataType acc_dtype) {
    MPSGraph* mpsGraph = cachedGraph->graph();
    MPSGraphTensor* primary = nil;
    MPSGraphTensor* secondary = nil;
    castInputsArithmeticOp(mpsGraph,
                           cachedGraph->primaryTensor, cachedGraph->secondaryTensor,
                           primary, secondary,
                           self_dtype, other_dtype, acc_dtype);
    MPSGraphTensor* divTensor =  [mpsGraph divisionWithPrimaryTensor:primary
                                                     secondaryTensor:secondary
                                                                name:nil];
    if (!rounding_mode.has_value()) {
      return divTensor;
    } else if (*rounding_mode == "trunc") {
      return trunc_tensor(mpsGraph, divTensor);
    } else if (*rounding_mode == "floor") {
      return [mpsGraph floorWithTensor:divTensor name:nil];
    }
    assert(0 && "Invalid rounding mode\n");
    return nullptr;
  };
  binaryOpTensor(self, other, Scalar(1.0), output, op_name + "_out_mps:" + (rounding_mode.has_value() ? c10::str(*rounding_mode) : ""), true, div_mode_op_block);
}

void add_sub_template(const Tensor& self, const Tensor& other, const Scalar& alpha, const Tensor& output, std::string op_name)
{
  if (alpha.toDouble() == 0.0)
    const_cast<Tensor&>(output) = self.clone();

  const bool alpha_has_value = alpha.toDouble() != 1.0;
  if (alpha_has_value) {
    auto commonDtype = at::result_type(self, other);
    at::native::alpha_check(commonDtype, alpha);
  }

  BinaryOpBlock add_sub_op_block = ^MPSGraphTensor* (BinaryOpCachedGraph* cachedGraph,
                                                     const MPSDataType self_dtype,
                                                     const MPSDataType other_dtype,
                                                     const MPSDataType acc_dtype) {
    MPSGraph* mpsGraph = cachedGraph->graph();
    MPSGraphTensor* secondaryTensor = cachedGraph->secondaryTensor;

    // if alpha is 1.0, then we don't bother adding another multiply to graph
    if (alpha_has_value) {
      cachedGraph->alphaTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(other.scalar_type()), @[@1]);
      secondaryTensor = [mpsGraph multiplicationWithPrimaryTensor:cachedGraph->secondaryTensor
                                                  secondaryTensor:cachedGraph->alphaTensor
                                                             name:nil];
    }

    MPSGraphTensor* primary = nil;
    MPSGraphTensor* secondary = nil;
    castInputsArithmeticOp(mpsGraph, 
                           cachedGraph->primaryTensor, secondaryTensor,
                           primary, secondary,
                           self_dtype, other_dtype, acc_dtype);

    if (op_name == "add")
      return [mpsGraph additionWithPrimaryTensor:primary
                                 secondaryTensor:secondary
                                            name:nil];
    else
      return [mpsGraph subtractionWithPrimaryTensor:primary
                                    secondaryTensor:secondary
                                               name:nil];
  };
  // add alpha's type to the key only if multiply was added to graph
  binaryOpTensor(self, other, alpha, output, op_name + "_out_mps:" + (alpha_has_value ? getMPSTypeString(alpha.type()) : ""), true, add_sub_op_block);
}

} // namespace mps

#define CREATE_MPS_BINARY_OP_FUNC(func_out, func_stub, other_type)                              \
TORCH_IMPL_FUNC(func_out) (const Tensor& self, const other_type& other, const Tensor& output) { \
  mps::binaryOp##other_type(self, other, Scalar(1.0), output, #func_stub, true,                 \
    ^MPSGraphTensor* (mps::BinaryOpCachedGraph* cachedGraph,                                    \
                      const MPSDataType self_dtype,                                             \
                      const MPSDataType other_dtype,                                            \
                      const MPSDataType acc_dtype) {                                            \
      MPSGraph* mpsGraph = cachedGraph->graph();                                                \
      MPSGraphTensor* primary = nil;                                                            \
      MPSGraphTensor* secondary = nil;                                                          \
      castInputsArithmeticOp(mpsGraph,                                                          \
                             cachedGraph->primaryTensor, cachedGraph->secondaryTensor,          \
                             primary, secondary,                                                \
                             self_dtype, other_dtype, acc_dtype);                               \
      return [mpsGraph func_stub##WithPrimaryTensor:primary                                     \
                                    secondaryTensor:secondary                                   \
                                               name:nil]; });                                   \
}

// Boolean Ops require casting output to "MPSDataTypeBool"
#define CREATE_MPS_BOOLEAN_OP_FUNC(func_out, func_stub, other_type)                                      \
TORCH_IMPL_FUNC(func_out) (const Tensor& self, const other_type& other, const Tensor& output) {          \
  mps::binaryOp##other_type(self, other, Scalar(1.0), output, #func_stub, false                          \
    ^MPSGraphTensor* (mps::BinaryOpCachedGraph* cachedGraph,                                             \
                      const MPSDataType self_dtype,                                                      \
                      const MPSDataType other_dtype,                                                     \
                      const MPSDataType acc_dtype) {                                                     \
      MPSGraph* mpsGraph = cachedGraph->graph();                                                         \
      MPSGraphTensor* primary = nil;                                                                     \
      MPSGraphTensor* secondary = nil;                                                                   \
      castInputsBooleanOp(mpsGraph,                                                                      \
                          cachedGraph->primaryTensor, cachedGraph->secondaryTensor,                      \
                          primary, secondary,                                                            \
                          self_dtype, other_dtype);                                                      \
      MPSGraphTensor* outputTensor = [mpsGraph func_stub##WithPrimaryTensor:primary                      \
                                                            secondaryTensor:secondary                    \
                                                                       name:nil];                        \
      return [mpsGraph castTensor:outputTensor toType:MPSDataTypeBool name:@"boolOut"]; });              \
}

// Boolean Binary Ops
CREATE_MPS_BOOLEAN_OP_FUNC(eq_scalar_out_mps, equal, Scalar);
CREATE_MPS_BOOLEAN_OP_FUNC(eq_tensor_out_mps, equal, Tensor);
CREATE_MPS_BOOLEAN_OP_FUNC(ne_scalar_out_mps, notEqual, Scalar);
CREATE_MPS_BOOLEAN_OP_FUNC(ne_tensor_out_mps, notEqual, Tensor);
CREATE_MPS_BOOLEAN_OP_FUNC(le_scalar_out_mps, lessThanOrEqualTo, Scalar);
CREATE_MPS_BOOLEAN_OP_FUNC(le_tensor_out_mps, lessThanOrEqualTo, Tensor);
CREATE_MPS_BOOLEAN_OP_FUNC(lt_scalar_out_mps, lessThan, Scalar);
CREATE_MPS_BOOLEAN_OP_FUNC(lt_tensor_out_mps, lessThan, Tensor);
CREATE_MPS_BOOLEAN_OP_FUNC(ge_scalar_out_mps, greaterThanOrEqualTo, Scalar);
CREATE_MPS_BOOLEAN_OP_FUNC(ge_tensor_out_mps, greaterThanOrEqualTo, Tensor);
CREATE_MPS_BOOLEAN_OP_FUNC(gt_scalar_out_mps, greaterThan, Scalar);
CREATE_MPS_BOOLEAN_OP_FUNC(gt_tensor_out_mps, greaterThan, Tensor);

// Arithmetic Binary Ops
CREATE_MPS_BINARY_OP_FUNC(minimum_out_mps, minimum, Tensor);
CREATE_MPS_BINARY_OP_FUNC(maximum_out_mps, maximum, Tensor);
CREATE_MPS_BINARY_OP_FUNC(mul_out_mps, multiplication, Tensor);
CREATE_MPS_BINARY_OP_FUNC(pow_tensor_scalar_out_mps, power, Scalar);
CREATE_MPS_BINARY_OP_FUNC(pow_tensor_tensor_out_mps, power, Tensor);
CREATE_MPS_BINARY_OP_FUNC(atan2_mps_out, atan2, Tensor);


TORCH_IMPL_FUNC(div_out_mode_mps) (const Tensor& self, const Tensor& other, c10::optional<c10::string_view> rounding_mode, const Tensor& output) {
  mps::div_mode_template(self, other, rounding_mode, output, "div_mode");
}

TORCH_IMPL_FUNC(div_out_mps) (const Tensor& self, const Tensor& other, const Tensor& output) {
  mps::div_mode_template(self, other, c10::nullopt, output, "div");
}

TORCH_IMPL_FUNC(add_out_mps) (const Tensor& self, const Tensor& other, const Scalar& alpha, const Tensor& output) {
  mps::add_sub_template(self, other, alpha, output, "add");
}

TORCH_IMPL_FUNC(sub_out_mps) (const Tensor& self, const Tensor& other, const Scalar& alpha, const Tensor& output) {
  mps::add_sub_template(self, other, alpha, output, "sub");
}


TORCH_IMPL_FUNC(logaddexp_out_mps) (const Tensor& self, const Tensor& other, const Tensor& output)
{
      using namespace mps;
      MPSStream* stream = getCurrentMPSStream();

      if (&output != &self) {
          output.resize_(self.sizes());;
      }

      // Derive from MPSCachedGraph
      struct CachedGraph : public MPSCachedGraph
      {
        CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
        MPSGraphTensor *inputTensor_ = nil;
        MPSGraphTensor *otherTensor_ = nil;
        MPSGraphTensor *outputTensor_ = nil;
      };

      MPSGraphCache* cache_ = MPSGraphCache::getInstance();

      @autoreleasepool {
        string key = "log_base_e_out_mps:" + getTensorsStringKey({self, other});
        CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));

        if(!cachedGraph) {
          MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {
            CachedGraph *newCachedGraph = nil;

            @autoreleasepool {
              MPSGraph* mpsGraph = make_mps_graph();
              newCachedGraph = new CachedGraph(mpsGraph);
              MPSGraphTensor* xTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
              MPSGraphTensor* yTensor = mpsGraphRankedPlaceHolder(mpsGraph, other);
              MPSGraphTensor* ePowXTensor = [mpsGraph exponentWithTensor:xTensor
                                                                         name:nil];
              MPSGraphTensor* ePowYTensor = [mpsGraph exponentWithTensor:yTensor
                                                                         name:nil];
              MPSGraphTensor* sumTensor = [mpsGraph additionWithPrimaryTensor:ePowXTensor
                                                                secondaryTensor:ePowYTensor
                                                                     name:nil];
              MPSGraphTensor* outputTensor = [mpsGraph logarithmWithTensor:sumTensor
                                                                     name:nil];

              newCachedGraph->inputTensor_ = xTensor;
              newCachedGraph->otherTensor_ = yTensor;
              newCachedGraph->outputTensor_ = outputTensor;
            }
            return newCachedGraph;
          });
          cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
        }

        Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
        Placeholder otherPlaceholder = Placeholder(cachedGraph->otherTensor_, other);
        Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);

        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
          selfPlaceholder.getMPSGraphTensor() : selfPlaceholder.getMPSGraphTensorData(),
          otherPlaceholder.getMPSGraphTensor() : otherPlaceholder.getMPSGraphTensorData()
        };
        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
          outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
        };

        runMPSGraph(stream, cachedGraph->graph(), feeds, results);
      }

    }

TORCH_IMPL_FUNC(logaddexp2_out_mps) (const Tensor& self, const Tensor& other, const Tensor& output)
{
      using namespace mps;
      MPSStream* stream = getCurrentMPSStream();

      if (&output != &self) {
          output.resize_(self.sizes());;
      }

      // Derive from MPSCachedGraph
      struct CachedGraph : public MPSCachedGraph
      {
        CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
        MPSGraphTensor *inputTensor_ = nil;
        MPSGraphTensor *otherTensor_ = nil;
        MPSGraphTensor *outputTensor_ = nil;
      };

      MPSGraphCache* cache_ = MPSGraphCache::getInstance();

      @autoreleasepool {
        string key = "log_base_two_out_mps:" + getTensorsStringKey({self, other});
        CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));

        if(!cachedGraph) {
          MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {
            CachedGraph *newCachedGraph = nil;

            @autoreleasepool {
              MPSGraph* mpsGraph = make_mps_graph();
              newCachedGraph = new CachedGraph(mpsGraph);
              MPSGraphTensor* xTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
              MPSGraphTensor* yTensor = mpsGraphRankedPlaceHolder(mpsGraph, other);
              MPSGraphTensor* twoPowXTensor = [mpsGraph exponentBase2WithTensor:xTensor
                                                                         name:nil];
              MPSGraphTensor* twoPowYTensor = [mpsGraph exponentBase2WithTensor:yTensor
                                                                         name:nil];
              MPSGraphTensor* sumTensor = [mpsGraph additionWithPrimaryTensor:twoPowXTensor
                                                                secondaryTensor:twoPowYTensor
                                                                     name:nil];
              MPSGraphTensor* outputTensor = [mpsGraph logarithmBase2WithTensor:sumTensor
                                                                     name:nil];

              newCachedGraph->inputTensor_ = xTensor;
              newCachedGraph->otherTensor_ = yTensor;
              newCachedGraph->outputTensor_ = outputTensor;
            }
            return newCachedGraph;
          });
          cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
        }

        Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
        Placeholder otherPlaceholder = Placeholder(cachedGraph->otherTensor_, other);
        Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);

        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
          selfPlaceholder.getMPSGraphTensor() : selfPlaceholder.getMPSGraphTensorData(),
          otherPlaceholder.getMPSGraphTensor() : otherPlaceholder.getMPSGraphTensorData()
        };
        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
          outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
        };

        runMPSGraph(stream, cachedGraph->graph(), feeds, results);
      }
}
} // namespace native
} // namespace at
