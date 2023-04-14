//  Copyright © 2022 Apple Inc.

#include <ATen/mps/MPSAllocatorInterface.h>
#include <ATen/native/mps/Copy.h>
#include <ATen/native/mps/operations/Scalar.h>
#include <ATen/Dispatch.h>
#include <ATen/mps/MPSProfiler.h>

namespace at::native {

Scalar _local_scalar_dense_mps(const Tensor& self) {
  Scalar r;

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
    at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16, self.scalar_type(), "_local_scalar_dense_mps", [&] {
      Tensor cpu_output = at::empty({1}, TensorOptions(at::CPU(self.scalar_type())));
      mps::mps_copy_(cpu_output, self, false);
      scalar_t cpu_scalar = *cpu_output.data_ptr<scalar_t>();
      r = Scalar(cpu_scalar);
   });

  return r;
}

namespace mps {

enum class ScalarOpTypes {
  UNSUPPORTED,
  // Primary Binary Ops
  ADD, SUB, DIV, MUL,
  // Comparison ops
  LT, LE, GT, GE, NE, EQ, LOGICAL_OR, LOGICAL_AND,
  // Bitwise Ops
  AND, OR, XOR,
};

// returns an enum associated with the passed string name along with
// whether the op is a comparison op or not.
static const std::pair<ScalarOpTypes, bool>
getScalarOpType(const std::string& op_name) {
  static std::unordered_map<std::string, std::pair<ScalarOpTypes, bool>> scalarOpsMap = {
    // Primary Binary Ops
    {"multiplication", {ScalarOpTypes::MUL, false}},
    {"div_out_mps:",   {ScalarOpTypes::DIV, false}},
    {"add_out_mps:",   {ScalarOpTypes::ADD, false}},
    {"sub_out_mps:",   {ScalarOpTypes::SUB, false}},

    // Bitwise ops
    {"and", {ScalarOpTypes::AND, false}},
    {"or" , {ScalarOpTypes::OR , false}},
    {"xor", {ScalarOpTypes::XOR, false}},

    // Comparison ops
    {"lessThan",             {ScalarOpTypes::LT,          true}},
    {"lessThanOrEqualTo",    {ScalarOpTypes::LE,          true}},
    {"greaterThan",          {ScalarOpTypes::GT,          true}},
    {"greaterThanOrEqualTo", {ScalarOpTypes::GE,          true}},
    {"notEqual",             {ScalarOpTypes::NE,          true}},
    {"equal",                {ScalarOpTypes::EQ,          true}},
    {"logicalOR",            {ScalarOpTypes::LOGICAL_OR,  true}},
    {"logicalAND",           {ScalarOpTypes::LOGICAL_AND, true}},
  };

  if (scalarOpsMap.count(op_name) == 0) {
    return {ScalarOpTypes::UNSUPPORTED, false};
  }
  return scalarOpsMap[op_name];
}

#define BINARY_OP_CASE(OP_TYPE, OPERATOR) \
  case ScalarOpTypes::OP_TYPE: \
    return (selfValue) OPERATOR (otherValue)

template<typename common_dtype, typename output_dtype>
static output_dtype scalar_binary_ops_impl(const ScalarOpTypes scalarOpType,
                                           const Scalar& selfScalar,
                                           const Scalar& otherScalar) {
  common_dtype selfValue  = selfScalar.to<common_dtype>();
  common_dtype otherValue = otherScalar.to<common_dtype>();

  switch (scalarOpType) {
    BINARY_OP_CASE(ADD, + );
    BINARY_OP_CASE(SUB, - );
    BINARY_OP_CASE(MUL, * );
    BINARY_OP_CASE(DIV, / );
    BINARY_OP_CASE(LT , < );
    BINARY_OP_CASE(LE , <=);
    BINARY_OP_CASE(GT , > );
    BINARY_OP_CASE(GE , >=);
    BINARY_OP_CASE(NE , !=);
    BINARY_OP_CASE(EQ , ==);
    BINARY_OP_CASE(LOGICAL_AND, &&);
    BINARY_OP_CASE(LOGICAL_OR , ||);

    default:
      AT_ERROR("Unknown scalar Binary operation: ", (uint32_t) scalarOpType);
  }
}

template<typename scalar_t>
static scalar_t scalar_bitwise_ops_impl(const ScalarOpTypes scalarOpType,
                                        const scalar_t* selfCpuPtr,
                                        const scalar_t* otherCpuPtr) {
  scalar_t selfValue  = *selfCpuPtr;
  scalar_t otherValue = *otherCpuPtr;

  switch (scalarOpType) {
    BINARY_OP_CASE(AND, &);
    BINARY_OP_CASE(OR , |);
    BINARY_OP_CASE(XOR, ^);

    default:
      AT_ERROR("Unknown scalar Binary operation: ", (uint32_t) scalarOpType);
  }
}

bool scalar_ops_mps(const std::string& op, const Tensor& self, const Tensor& other,
                    const Tensor& output, ScalarOpCategories scalarOpCategory) {
  const auto& allocator = *at::mps::getIMPSAllocator();
  if (!allocator.isSharedStorageSupported()) {
    return false;
  }
  if (self.dim() > 0 || other.dim() > 0) {
    return false;
  }
  if (!self.is_contiguous() || !other.is_contiguous() || !output.is_contiguous()) {
    return false;
  }

  ScalarOpTypes scalarOpType = ScalarOpTypes::UNSUPPORTED;
  bool isComparisonOp = false;
  std::tie(scalarOpType, isComparisonOp) = getScalarOpType(op);
  // fallback to Binary Ops kernels/graphs if scalar op isn't supported
  if (scalarOpType == ScalarOpTypes::UNSUPPORTED) {
    return false;
  }

  void* selfCpuPtr = self.storage().data();
  void* otherCpuPtr = other.storage().data();
  void* outputCpuPtr = output.storage().data();
  uint32_t selfOffset = self.storage_offset() * self.element_size();
  uint32_t otherOffset = other.storage_offset() * other.element_size();

  ScalarType selfDType = self.scalar_type();
  ScalarType otherDType = other.scalar_type();
  ScalarType outputDType = output.scalar_type();

  ScalarType commonDType = c10::promoteTypes(selfDType, otherDType);
  if (isIntegralType(commonDType, true)) {
    // integer inputs must be cast to float, if output is float
    if (isFloatingType(outputDType)) {
      commonDType = outputDType;
    // in boolean comparison ops with signed vs. unsigned integers, we always cast to the unsigned type
    } else if (outputDType == ScalarType::Bool &&
              (selfDType == ScalarType::Byte ||
              otherDType == ScalarType::Byte)) {
      commonDType = ScalarType::Byte;
    }
  }
  // double type for CPU scalars must be downcast to be compatible with MPS
  if (commonDType == kDouble) {
    if (!isComparisonOp) {
      commonDType = outputDType;
    } else if (selfDType != kDouble) {
      commonDType = selfDType;
    } else if (otherDType != kDouble) {
      commonDType = otherDType;
    }
  }

  // note that if the retainCount > 1 for any of the input/output buffers, then there's a
  // dependency on GPU timeline, and we cannot do the computation on CPU timeline in this method.
  uint32_t retainCount = 0;

  if (self.is_mps()) {
    std::tie(selfCpuPtr, retainCount) = allocator.getSharedBufferPtr(selfCpuPtr, true);
    if (!selfCpuPtr || retainCount > 1) {
      return false;
    }
    selfCpuPtr = (char*) selfCpuPtr + selfOffset;
  }
  if (other.is_mps()) {
    std::tie(otherCpuPtr, retainCount) = allocator.getSharedBufferPtr(otherCpuPtr, true);
    if (!otherCpuPtr || retainCount > 1) {
      return false;
    }
    otherCpuPtr = (char*) otherCpuPtr + otherOffset;
  }
  std::tie(outputCpuPtr, retainCount) = allocator.getSharedBufferPtr(outputCpuPtr, true);
  if (!outputCpuPtr || retainCount > 1) {
    return false;
  }

  auto& profiler = getMPSProfiler();
  const std::string& profiler_name = "scalar_" + op;
  const bool isOperationProfilingEnabled = profiler.isCPUFallbackProfilingEnabled();
  if (isOperationProfilingEnabled) {
    profiler.beginProfileCPUFallback(profiler_name, {self, other, output});
  }

  if (scalarOpCategory == ScalarOpCategories::BINARY_OPS) {
    Scalar selfScalar, otherScalar;
    AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBool, selfDType, "scalar_self_cast", [&]() {
      selfScalar = Scalar(*(static_cast<scalar_t*>(selfCpuPtr)));
    });
    AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBool, otherDType, "scalar_other_cast", [&]() {
      otherScalar = Scalar(*(static_cast<scalar_t*>(otherCpuPtr)));
    });

    AT_DISPATCH_ALL_TYPES_AND2(kHalf, kBool, commonDType, "scalar_binary_ops", [&]() {
      if (isComparisonOp) {
        *((bool*)outputCpuPtr) =
            scalar_binary_ops_impl<scalar_t, bool>(scalarOpType, selfScalar, otherScalar);
      } else {
        *((scalar_t*)outputCpuPtr) =
            scalar_binary_ops_impl<scalar_t, scalar_t>(scalarOpType, selfScalar, otherScalar);
      }
    });

  // bitwise ops only support integral types
  } else if (scalarOpCategory == ScalarOpCategories::BITWISE_OPS) {
    AT_DISPATCH_INTEGRAL_TYPES_AND(kBool, commonDType, "scalar_bitwise_ops", [&]() {
      *((scalar_t*)outputCpuPtr) = scalar_bitwise_ops_impl<scalar_t>(scalarOpType,
                                        static_cast<scalar_t*>(selfCpuPtr),
                                        static_cast<scalar_t*>(otherCpuPtr));
    });
  } else {
    AT_ERROR("Unsupported scalar operation type: ", (uint32_t) scalarOpCategory);
  }

  if (isOperationProfilingEnabled) {
    profiler.endProfileCPUFallback(profiler_name);
  }

  return true;
}

} // namespace mps
} // namespace at::native
