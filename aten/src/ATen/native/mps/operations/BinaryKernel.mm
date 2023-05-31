#include <ATen/native/mps/OperationUtils.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/mps/IndexKernels.h>

namespace at::native {
namespace mps {

enum class BinaryKernelType {
  Scalar,
  LHS_Scalar,
  RHS_Scalar,
  Tensor,
  Strided_LHS_Scalar,
  Strided_RHS_Scalar,
  Strided_Tensor
};

static char* BINARY_OP_TEMPLATE_TENSOR = R"METAL_BINARY(
kernel void {3}_kernel(uint tid                   [[thread_position_in_grid]],
                       const device {1} * input   [[buffer(0)]],
                       const device {2} * other   [[buffer(1)]],
                       device       {0} * output  [[buffer(2)]]) {{
  output[tid] = ({5})input[tid] {4} ({5})other[tid];
}}
)METAL_BINARY";

static char* BINARY_OP_TEMPLATE_STRIDED_TENSOR = GET_IDX_TEMPLATE
R"METAL_BINARY(
kernel void {3}_kernel_strided(uint tid [[thread_position_in_grid]],
                       const device void     * input_             [[buffer(0)]],
                       const device void     * other_             [[buffer(1)]],
                       device void           * output_            [[buffer(2)]],
                       constant uint         * iter_shape         [[buffer(3)]],
                       constant uint         & num_dimensions     [[buffer(4)]],
                       constant packed_uint3 * strides            [[buffer(5)]]) {{
  uint3 offsets = get_idx(tid, iter_shape, num_dimensions, strides);

  device {0}* output       = (device {0}*)((device uint8_t*)output_  + offsets.x);
  const device {1}* input  = (const device {1}*)((const device uint8_t*)input_ + offsets.y);
  const device {2}* other  = (const device {2}*)((const device uint8_t*)other_ + offsets.z);

  *output = ({5})*input {4} ({5})*other;
}}
)METAL_BINARY";

static  char* BINARY_OP_TEMPLATE_LHS_SCALAR = R"METAL_BINARY(
kernel void {3}_kernel_scalar_lhs(uint tid               [[thread_position_in_grid]],
                              const device {1} & input   [[buffer(0)]],
                              const device {2} * other   [[buffer(1)]],
                              device       {0} * output  [[buffer(2)]]) {{
  output[tid] = ({5})input {4} ({5})other[tid];
}}
)METAL_BINARY";

static  char* BINARY_OP_TEMPLATE_RHS_SCALAR = R"METAL_BINARY(
kernel void {3}_kernel_scalar_rhs(uint tid  [[thread_position_in_grid]],
                              const device {1}  * input   [[buffer(0)]],
                              const device {2}  & other   [[buffer(1)]],
                              device       {0}  * output  [[buffer(2)]]) {{
  output[tid] = ({5})input[tid] {4} ({5})other;
}}
)METAL_BINARY";

static  char* BINARY_OP_TEMPLATE_SCALAR = R"METAL_BINARY(
kernel void {3}_kernel_scalar(uint tid                       [[thread_position_in_grid]],
                              const device {1} & input       [[buffer(0)]],
                              const device {2} & other       [[buffer(1)]],
                              device       {0} & output      [[buffer(2)]]) {{
  output = ({5})input {4} ({5})other;
}}
)METAL_BINARY";

static  char* BINARY_OP_TEMPLATE_STRIDED_RHS_SCALAR = GET_IDX_TEMPLATE
R"METAL_BINARY(
kernel void {3}_kernel_scalar_rhs_strided(uint tid               [[thread_position_in_grid]],
                       const device void     * input_            [[buffer(0)]],
                       const device {2}      & other             [[buffer(1)]],
                       device void           * output_           [[buffer(2)]],
                       constant uint         * iter_shape        [[buffer(3)]],
                       constant uint         & num_dimensions    [[buffer(4)]],
                       constant packed_uint3 * strides           [[buffer(5)]]) {{
  uint3 offsets = get_idx(tid, iter_shape, num_dimensions, strides);

  device {0}* output = (device {0}*)((device uint8_t*)output_ + offsets.x);
  const device {1}* input = (const device {1}*)((const device uint8_t*)input_ + offsets.y);

  *output = ({5})*input {4} ({5})other;
}}
)METAL_BINARY";

static  char* BINARY_OP_TEMPLATE_STRIDED_LHS_SCALAR = GET_IDX_TEMPLATE
R"METAL_BINARY(
kernel void {3}_kernel_scalar_lhs_strided(uint tid               [[thread_position_in_grid]],
                       const device {1}      & input             [[buffer(0)]],
                       const device void     * other_            [[buffer(1)]],
                       device void           * output_           [[buffer(2)]],
                       constant uint         * iter_shape        [[buffer(3)]],
                       constant uint         & num_dimensions    [[buffer(4)]],
                       constant packed_uint3 * strides           [[buffer(5)]]) {{
  uint3 offsets = get_idx(tid, iter_shape, num_dimensions, strides);

  device {0}* output = (device {0}*)((device uint8_t*)output_ + offsets.x);
  const device {2}* other = (const device {2}*)((const device uint8_t*)other_ + offsets.z);

  *output = ({5})input {4} ({5})*other;
}}
)METAL_BINARY";

static id<MTLLibrary> compileBinaryOpsLibrary(id<MTLDevice> device,
                                              const std::string& t1,
                                              const std::string& t2,
                                              const std::string& t3,
                                              const std::string& common_dtype,
                                              const std::string& op,
                                              const std::string& kernel_operator,
                                              BinaryKernelType binaryKernelType) {
  auto key = op + t1 + t2 + t3 + common_dtype + std::to_string(int(binaryKernelType));
  static std::unordered_map<std::string, id<MTLLibrary>> libMap;
  auto it = libMap.find(key);
  if (it != libMap.end()) {
    return it->second;
  }
  NSError *error = nil;
  MTLCompileOptions *options = [[MTLCompileOptions new] autorelease];
  MTLLanguageVersion languageVersion = MTLLanguageVersion2_2;
#if defined(__MAC_13_0)
  if (is_macos_13_or_newer(MacOSVersion::MACOS_VER_13_0_PLUS)) {
    languageVersion = MTLLanguageVersion3_0;
  }
#endif

  [options setLanguageVersion: languageVersion];
  char *str = nil;
  switch (binaryKernelType){
    case BinaryKernelType::Scalar:
      str = BINARY_OP_TEMPLATE_SCALAR;
      break;
    case BinaryKernelType::LHS_Scalar:
      str = BINARY_OP_TEMPLATE_LHS_SCALAR;
      break;
    case BinaryKernelType::RHS_Scalar:
      str = BINARY_OP_TEMPLATE_RHS_SCALAR;
      break;
    case BinaryKernelType::Tensor:
      str = BINARY_OP_TEMPLATE_TENSOR;
      break;
    case BinaryKernelType::Strided_Tensor:
      str = BINARY_OP_TEMPLATE_STRIDED_TENSOR;
      break;
    case BinaryKernelType::Strided_LHS_Scalar:
      str = BINARY_OP_TEMPLATE_STRIDED_LHS_SCALAR;
      break;
    case BinaryKernelType::Strided_RHS_Scalar:
      str = BINARY_OP_TEMPLATE_STRIDED_RHS_SCALAR;
      break;
    default:
      TORCH_CHECK(false, "Unknown binary template");
  }

  auto rc = [device newLibraryWithSource:[NSString stringWithUTF8String:fmt::format(str, t1, t2, t3, op, kernel_operator, common_dtype).c_str()]
                                 options:options
                                   error:&error];
  TORCH_CHECK(rc != nil && error == nil, "Failed to compile library: ", [[error localizedDescription] UTF8String]);
  libMap[key] = rc;
  return rc;
}

static id<MTLComputePipelineState> getBinaryPSO(id<MTLDevice> device,
                                                const std::string& t1,
                                                const std::string& t2,
                                                const std::string& t3,
                                                const std::string& common_dtype,
                                                const std::string& fname,
                                                const std::string& op,
                                                const std::string& kernel_operator,
                                                BinaryKernelType binaryKernelType) {
  auto key = t1 + t2 + t3 + common_dtype + fname;
  static std::unordered_map<std::string, id<MTLComputePipelineState>> cplMap;
  auto it = cplMap.find(key);
  if (it != cplMap.end()) {
     return it->second;
  }
  NSError *error = nil;
  auto library = compileBinaryOpsLibrary(device, t1, t2, t3, common_dtype, op, kernel_operator, binaryKernelType);
  id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithUTF8String:fname.c_str()]];
  TORCH_CHECK(func != nil, "Can't get function ", fname);
  auto rc = [device newComputePipelineStateWithFunction:func error:&error];
  TORCH_CHECK(rc != nil && error == nil, "Failed to construct pipeline state: ", [[error localizedDescription] UTF8String]);
  cplMap[key] = rc;
  return rc;
}

static
void dispatch_binary_kernel_mps_(TensorIteratorBase& iter, const std::string& op, const std::string& kernel_operator) {
  Tensor inputTensor;
  Tensor otherTensor;
  BinaryKernelType type;

  int scalar_pos = 0;
  bool all_scalar = false;
  const Tensor& outputTensor = iter.tensor(0);
  inputTensor = iter.tensor(1);
  otherTensor = iter.tensor(2);

  if (inputTensor.scalar_type() == kDouble) {
    inputTensor = inputTensor.to(iter.common_dtype());
  }
  if (otherTensor.scalar_type() == kDouble) {
    otherTensor = otherTensor.to(iter.common_dtype());
  }

  auto outputDataType = outputTensor.scalar_type();
  auto inputDataType = inputTensor.scalar_type();
  auto otherDataType = otherTensor.scalar_type();
  ScalarType common_dtype = iter.common_dtype();
  if (isIntegralType(common_dtype, true)) {
    // integer inputs must be cast to float, if output is float
    if (isFloatingType(outputDataType)) {
      common_dtype = outputDataType;
    // in boolean comparison ops with signed vs. unsigned integers, we always cast to the unsigned type
    } else if (outputDataType == ScalarType::Bool &&
              (inputDataType == ScalarType::Byte ||
                otherDataType == ScalarType::Byte)) {
      common_dtype = ScalarType::Byte;
    }
  }

  // workaround for bool issues (e.g. bool dtype: true + true in Metal would be 0, but the expected result is still 1 in PyTorch)
  if (outputDataType == kBool && (inputDataType == kByte || otherDataType == kByte)) {
    inputDataType = otherDataType = kByte;
  } else {
    if (inputDataType == kBool) {
      inputDataType = kChar;
    }
    if (otherDataType == kBool) {
      otherDataType = kChar;
    }
  }

  if (iter.tensor(1).numel() == 1 && iter.tensor(2).numel() == 1) {
    all_scalar = true;
  } else if (iter.tensor(1).numel() == 1) {
    scalar_pos = 1;
  } else if (iter.tensor(2).numel() == 1) {
    scalar_pos = 2;
  }

  if (!scalar_pos && !all_scalar) {
    std::vector<Tensor> tmp = expand_outplace({inputTensor, otherTensor});
    inputTensor = tmp[0];
    otherTensor = tmp[1];
  }

  if (inputTensor.numel() == 0 || otherTensor.numel() == 0) {
    return;
  }

  bool allContiguous = false;
  if (inputTensor.is_contiguous() && otherTensor.is_contiguous() && outputTensor.is_contiguous()) {
    allContiguous = true;
  }

  MPSStream* mpsStream = getCurrentMPSStream();
  id<MTLDevice> device = MPSDevice::getInstance()->device();

  id<MTLBuffer> inputBuffer = mps::getMTLBufferStorage(inputTensor);
  id<MTLBuffer> otherBuffer = mps::getMTLBufferStorage(otherTensor);
  id<MTLBuffer> outputBuffer = mps::getMTLBufferStorage(outputTensor);
  uint32_t inputTensorStorage = inputTensor.storage_offset() * inputTensor.element_size();
  uint32_t otherTensorStorage = otherTensor.storage_offset() * otherTensor.element_size();
  mps::MPSScalar scalar;
  if (all_scalar) {
    type = BinaryKernelType::Scalar;
    if (iter.is_cpu_scalar(1)) {
      scalar = mps::getMPSScalar(inputTensor.item(), inputTensor.scalar_type());
      inputBuffer = (id<MTLBuffer>)getIMPSAllocator()->allocScalarBufferWithValue(&scalar.value, scalar.size).get();
      inputTensorStorage = 0;
    }
    if (iter.is_cpu_scalar(2)) {
      scalar = mps::getMPSScalar(otherTensor.item(), otherTensor.scalar_type());
      otherBuffer = (id<MTLBuffer>)getIMPSAllocator()->allocScalarBufferWithValue(&scalar.value, scalar.size).get();
      otherTensorStorage = 0;
    }
  } else if (scalar_pos) {
    if (allContiguous) {
      type = scalar_pos == 1 ? BinaryKernelType::LHS_Scalar : BinaryKernelType::RHS_Scalar;
    } else {
      type = scalar_pos == 1 ? BinaryKernelType::Strided_LHS_Scalar : BinaryKernelType::Strided_RHS_Scalar;
    }

    if (iter.is_cpu_scalar(scalar_pos)) {
      if (scalar_pos == 1) {
        scalar = mps::getMPSScalar(inputTensor.item(), inputTensor.scalar_type());
        inputBuffer = (id<MTLBuffer>)getIMPSAllocator()->allocScalarBufferWithValue(&scalar.value, scalar.size).get();
        inputTensorStorage = 0;
      } else {
        scalar = mps::getMPSScalar(otherTensor.item(), otherTensor.scalar_type());
        otherBuffer = (id<MTLBuffer>)getIMPSAllocator()->allocScalarBufferWithValue(&scalar.value, scalar.size).get();
        otherTensorStorage = 0;
      }
    }
  } else {
    type = allContiguous ? BinaryKernelType::Tensor : BinaryKernelType::Strided_Tensor;
  }

  const uint32_t nDim = iter.ndim();
  constexpr uint32_t nOffsets = 3;

  dispatch_sync(mpsStream->queue(), ^(){
    @autoreleasepool {
      uint32_t numThreads = iter.numel();
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      MTLSize gridSize = MTLSizeMake(numThreads, 1, 1);
      const IntArrayRef& iterShape = iter.shape();
      std::vector<uint32_t> iterShapeData(iterShape.size());
      std::vector<std::array<uint32_t, nOffsets>> strides(nDim);

      if (!allContiguous) {
        for (const auto i: c10::irange(iterShape.size())) {
          TORCH_CHECK(i <= UINT32_MAX);
          iterShapeData[i] = (uint32_t)(iterShape[i]);
        }

        for (const auto i: c10::irange(nDim)) {
          for (const auto offset: c10::irange(nOffsets)) {
              strides[i][offset] = iter.strides(offset)[i];
          }
        }
      }

      std::string kernel = op;
      kernel += "_kernel";
      if (all_scalar) {
        kernel += "_scalar";
      }
      if (scalar_pos) {
        kernel += "_scalar_";
        if (scalar_pos == 1) {
          kernel += "lhs";
        } else {
          kernel += "rhs";
        }
      }
      if (!allContiguous) {
        kernel += "_strided";
      }

      id<MTLComputePipelineState> binaryPSO = mps::getBinaryPSO(device,
                                                          getMetalScalarType(outputDataType),
                                                          getMetalScalarType(inputDataType),
                                                          getMetalScalarType(otherDataType),
                                                          getMetalScalarType(common_dtype),
                                                          kernel,
                                                          op,
                                                          kernel_operator,
                                                          type);
      getMPSProfiler().beginProfileKernel(binaryPSO, kernel, {inputTensor, otherTensor, outputTensor});
      [computeEncoder setComputePipelineState:binaryPSO];
      [computeEncoder setBuffer:inputBuffer  offset:inputTensorStorage atIndex:0];
      [computeEncoder setBuffer:otherBuffer  offset:otherTensorStorage atIndex:1];
      [computeEncoder setBuffer:outputBuffer offset:outputTensor.storage_offset() * outputTensor.element_size() atIndex:2];
      if (!allContiguous) {
        [computeEncoder setBytes:iterShapeData.data() length:sizeof(uint32_t) * iterShape.size() atIndex:3];
        [computeEncoder setBytes:&nDim length:sizeof(uint32_t) atIndex:4];
        [computeEncoder setBytes:strides.data() length:sizeof(uint32_t) * nDim * nOffsets atIndex:5];
      }

      NSUInteger tgSize = binaryPSO.maxTotalThreadsPerThreadgroup;
      if (tgSize > numThreads) {
          tgSize = numThreads;
      }

      MTLSize threadGroupSize = MTLSizeMake(tgSize, 1, 1);
      [computeEncoder dispatchThreads: gridSize
                threadsPerThreadgroup: threadGroupSize];
      mpsStream->commitAdaptive({inputTensor, otherTensor}, outputTensor, binaryPSO);
    }
  });
}

static char* BINARY_OP_TEMPLATE_STRIDED_TENSOR2 =
R"METAL_BINARY(
kernel void {3}_kernel_strided_1(uint linear_index [[thread_position_in_grid]],
                                 const device void     * input_             [[buffer(0)]],
                                 const device void     * other_             [[buffer(1)]],
                                 device void           * output_            [[buffer(2)]],

                                 constant int & size_input                  [[buffer(3)]],
                                 constant int & stride_input                [[buffer(4)]],

                                 constant int & size_other                  [[buffer(5)]],
                                 constant int & stride_other                [[buffer(6)]],

                                 constant int & size_output                 [[buffer(7)]],
                                 constant int & stride_output               [[buffer(8)]]) {{
  device {0}* output       = (device {0}*)(output_);
  const device {1}* input  = (const device {1}*)(input_);
  const device {2}* other  = (const device {2}*)(other_);

  // Input
  const int local_index_input = linear_index % size_input;
  const int strided_index_input = local_index_input * stride_input;

  // Other
  const int local_index_other = linear_index % size_other;
  const int strided_index_other = local_index_other * stride_other;

  // Output
  const int local_index_output = linear_index % size_output;
  const int strided_index_output = local_index_output * stride_output;

  output[strided_index_output] = ({5})input[strided_index_input] {4} ({5})other[strided_index_other];
}}

kernel void {3}_kernel_strided_2(uint linear_index [[thread_position_in_grid]],
                                 const device void     * input_             [[buffer(0)]],
                                 const device void     * other_             [[buffer(1)]],
                                 device void           * output_            [[buffer(2)]],

                                 constant packed_uint2 & size_input         [[buffer(3)]],
                                 constant packed_uint2 & stride_input       [[buffer(4)]],

                                 constant packed_uint2 & size_other                  [[buffer(5)]],
                                 constant packed_uint2 & stride_other                [[buffer(6)]],

                                 constant packed_uint2 & size_output                 [[buffer(7)]],
                                 constant packed_uint2 & stride_output               [[buffer(8)]]) {{
  device {0}* output       = (device {0}*)(output_);
  const device {1}* input  = (const device {1}*)(input_);
  const device {2}* other  = (const device {2}*)(other_);

  packed_uint2 local_index;

  // Input
  local_index.x = linear_index / size_input[1] % size_input[0];
  local_index.y = linear_index % size_input[1];
  const packed_uint2 strided_index_input = local_index * stride_input;

  // Other
  local_index.x = linear_index / size_other[1] % size_other[0];
  local_index.y = linear_index % size_other[1];
  const packed_uint2 strided_index_other = local_index * stride_other;

  // Output
  local_index.x = linear_index / size_output[1] % size_output[0];
  local_index.y = linear_index % size_output[1];
  const packed_uint2 strided_index_output = local_index * stride_output;

  output[strided_index_output.x + strided_index_output.y] = ({5})input[strided_index_input.x + strided_index_input.y] {4} ({5})other[strided_index_other.x + strided_index_other.y];
}}

kernel void {3}_kernel_strided_3(uint linear_index [[thread_position_in_grid]],
                                 const device void     * input_             [[buffer(0)]],
                                 const device void     * other_             [[buffer(1)]],
                                 device void           * output_            [[buffer(2)]],

                                 constant packed_uint3 & size_input         [[buffer(3)]],
                                 constant packed_uint3 & stride_input       [[buffer(4)]],

                                 constant packed_uint3 & size_other                  [[buffer(5)]],
                                 constant packed_uint3 & stride_other                [[buffer(6)]],

                                 constant packed_uint3 & size_output                 [[buffer(7)]],
                                 constant packed_uint3 & stride_output               [[buffer(8)]]) {{
  device {0}* output       = (device {0}*)(output_);
  const device {1}* input  = (const device {1}*)(input_);
  const device {2}* other  = (const device {2}*)(other_);

  packed_uint3 local_index;

  // Input
  local_index.x = linear_index / (size_input[2] * size_input[1]) % size_input[0];
  local_index.y = linear_index / size_input[2] % size_input[1];
  local_index.z = linear_index % size_input[2];
  const packed_uint3 strided_index_input = local_index * stride_input;

  // Other
  local_index.x = linear_index / (size_other[2] * size_other[1]) % size_other[0];
  local_index.y = linear_index / size_other[2] % size_other[1];
  local_index.z = linear_index % size_other[2];
  const packed_uint3 strided_index_other = local_index * stride_other;

  // Output
  local_index.x = linear_index / (size_output[2] * size_output[1]) % size_output[0];
  local_index.y = linear_index / size_output[2] % size_output[1];
  local_index.z = linear_index % size_output[2];
  const packed_uint3 strided_index_output = local_index * stride_output;

  output[strided_index_output.x + strided_index_output.y + strided_index_output.z] = ({5})input[strided_index_input.x + strided_index_input.y + strided_index_input.z] {4} ({5})other[strided_index_other.x + strided_index_other.y + strided_index_other.z];
}}

kernel void {3}_kernel_strided_4(uint linear_index [[thread_position_in_grid]],
                                 const device void     * input_             [[buffer(0)]],
                                 const device void     * other_             [[buffer(1)]],
                                 device void           * output_            [[buffer(2)]],

                                 constant packed_uint4 & size_input         [[buffer(3)]],
                                 constant packed_uint4 & stride_input       [[buffer(4)]],

                                 constant packed_uint4 & size_other                  [[buffer(5)]],
                                 constant packed_uint4 & stride_other                [[buffer(6)]],

                                 constant packed_uint4 & size_output                 [[buffer(7)]],
                                 constant packed_uint4 & stride_output               [[buffer(8)]]) {{
  device {0}* output       = (device {0}*)(output_);
  const device {1}* input  = (const device {1}*)(input_);
  const device {2}* other  = (const device {2}*)(other_);

  packed_uint4 local_index;

  // Input
  local_index.x = linear_index / (size_input[3] * size_input[2] * size_input[1]) % size_input[0];
  local_index.y = linear_index / (size_input[3] * size_input[2]) % size_input[1];
  local_index.z = linear_index / size_input[3] % size_input[2];
  local_index.w = linear_index % size_input[3];
  const packed_uint4 strided_index_input = local_index * stride_input;

  // Other
  local_index.x = linear_index / (size_other[3] * size_other[2] * size_other[1]) % size_other[0];
  local_index.y = linear_index / (size_other[3] * size_other[2]) % size_other[1];
  local_index.z = linear_index / size_other[3] % size_other[2];
  local_index.w = linear_index % size_other[3];
  const packed_uint4 strided_index_other = local_index * stride_other;

  // Output
  local_index.x = linear_index / (size_output[3] * size_output[2] * size_output[1]) % size_output[0];
  local_index.y = linear_index / (size_output[3] * size_output[2]) % size_output[1];
  local_index.z = linear_index / size_output[3] % size_output[2];
  local_index.w = linear_index % size_output[3];
  const packed_uint4 strided_index_output = local_index * stride_output;

  output[strided_index_output.x + strided_index_output.y + strided_index_output.z + strided_index_output.w] = ({5})input[strided_index_input.x + strided_index_input.y + strided_index_input.z + strided_index_input.w] {4} ({5})other[strided_index_other.x + strided_index_other.y + strided_index_other.z + strided_index_other.w];
}}

)METAL_BINARY";

static id<MTLLibrary> compileBinaryOpsLibrary2(id<MTLDevice> device,
                                              const std::string& t1,
                                              const std::string& t2,
                                              const std::string& t3,
                                              const std::string& common_dtype,
                                              const std::string& op,
                                              const std::string& kernel_operator) {
  auto key = op + t1 + t2 + t3 + common_dtype;
  static std::unordered_map<std::string, id<MTLLibrary>> libMap;
  auto it = libMap.find(key);
  if (it != libMap.end()) {
    return it->second;
  }
  NSError *error = nil;
  MTLCompileOptions *options = [[MTLCompileOptions new] autorelease];
  MTLLanguageVersion languageVersion = MTLLanguageVersion2_2;
#if defined(__MAC_13_0)
  if (is_macos_13_or_newer(MacOSVersion::MACOS_VER_13_0_PLUS)) {
    languageVersion = MTLLanguageVersion3_0;
  }
#endif

  [options setLanguageVersion: languageVersion];
  string s = fmt::format(BINARY_OP_TEMPLATE_STRIDED_TENSOR2, t1, t2, t3, op, kernel_operator, common_dtype);
  // std::cout << s << std::endl;
  auto rc = [device newLibraryWithSource:[NSString stringWithUTF8String:s.c_str()]
                                 options:options
                                   error:&error];
  TORCH_CHECK(rc != nil && error == nil, "Failed to compile library: ", [[error localizedDescription] UTF8String]);
  libMap[key] = rc;
  return rc;
}


static id<MTLComputePipelineState> getBinaryPSO2(id<MTLDevice> device,
                                                const std::string& t1,
                                                const std::string& t2,
                                                const std::string& t3,
                                                const std::string& common_dtype,
                                                const std::string& fname,
                                                const std::string& op,
                                                const std::string& kernel_operator) {
  auto key = t1 + t2 + t3 + common_dtype + fname;
  static std::unordered_map<std::string, id<MTLComputePipelineState>> cplMap;
  auto it = cplMap.find(key);
  if (it != cplMap.end()) {
     return it->second;
  }
  NSError *error = nil;
  auto library = compileBinaryOpsLibrary2(device, t1, t2, t3, common_dtype, op, kernel_operator);
  id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithUTF8String:fname.c_str()]];
  TORCH_CHECK(func != nil, "Can't get function ", fname);
  auto rc = [device newComputePipelineStateWithFunction:func error:&error];
  TORCH_CHECK(rc != nil && error == nil, "Failed to construct pipeline state: ", [[error localizedDescription] UTF8String]);
  cplMap[key] = rc;
  return rc;
}

static
bool dispatch_binary_kernel_mps(const Tensor& self, const Tensor& other, const Tensor& output, const std::string& op, const std::string& kernel_operator) {
  MPSStream* mpsStream = getCurrentMPSStream();
  if (!self.is_contiguous() || !other.is_contiguous() || !output.is_contiguous()) {
    if (self.numel() != 1 && other.numel() != 1 && output.numel() != 1) {
      if (self.numel() == other.numel()) {
        dispatch_sync(mpsStream->queue(), ^(){
          @autoreleasepool {
            auto inputDataType = self.scalar_type();
            auto otherDataType = other.scalar_type();
            auto outputDataType = output.scalar_type();
            ScalarType common_dtype = c10::promoteTypes(inputDataType, otherDataType);

            if (isIntegralType(common_dtype, true)) {
                // integer inputs must be cast to float, if output is float
                if (isFloatingType(outputDataType)) {
                  common_dtype = outputDataType;
                // in boolean comparison ops with signed vs. unsigned integers, we always cast to the unsigned type
                } else if (outputDataType == ScalarType::Bool &&
                          (inputDataType == ScalarType::Byte ||
                            otherDataType == ScalarType::Byte)) {
                  common_dtype = ScalarType::Byte;
                }
              }

              // workaround for bool issues (e.g. bool dtype: true + true in Metal would be 0, but the expected result is still 1 in PyTorch)
              if (outputDataType == kBool && (inputDataType == kByte || otherDataType == kByte)) {
                inputDataType = otherDataType = kByte;
              } else {
                if (inputDataType == kBool) {
                  inputDataType = kChar;
                }
                if (otherDataType == kBool) {
                  otherDataType = kChar;
                }
            }

            id<MTLBuffer> inputBuffer  = getMTLBufferStorage(self);
            id<MTLBuffer> otherBuffer  = getMTLBufferStorage(other);
            id<MTLBuffer> outputBuffer = getMTLBufferStorage(output);

            uint32_t kernel_size = self.sizes().size();
            std::vector<uint32_t> src_sizes(kernel_size == 0 ? 1 : kernel_size);
            std::vector<uint32_t> src_strides(kernel_size == 0 ? 1 : kernel_size);
            std::vector<uint32_t> other_sizes(kernel_size == 0 ? 1 : kernel_size);
            std::vector<uint32_t> other_strides(kernel_size == 0 ? 1 : kernel_size);
            std::vector<uint32_t> output_sizes(kernel_size == 0 ? 1 : kernel_size);
            std::vector<uint32_t> output_strides(kernel_size == 0 ? 1 : kernel_size);

            if (kernel_size == 0) {
              src_sizes[0] = src_strides[0] = other_sizes[0] = other_strides[0] = output_sizes[0] = output_strides[0] = 1;
            } else {
              for (int i = 0; i < kernel_size; i++) {
                src_sizes[i] = (uint32_t)(self.sizes()[i]);
                src_strides[i] = (uint32_t)(self.strides()[i]);

                other_sizes[i] = (uint32_t)(other.sizes()[i]);
                other_strides[i] = (uint32_t)(other.strides()[i]);

                output_sizes[i] = (uint32_t)(output.sizes()[i]);
                output_strides[i] = (uint32_t)(output.strides()[i]);
              }
            }

            std::string kernel = op;
            kernel += "_kernel";
            kernel += "_strided";
            kernel += "_" + std::to_string(kernel_size);
            id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
            id<MTLDevice> device = MPSDevice::getInstance()->device();
            id<MTLComputePipelineState> binaryPSO = getBinaryPSO2(device,
                          getMetalScalarType(outputDataType),
                          getMetalScalarType(inputDataType),
                          getMetalScalarType(otherDataType),
                          getMetalScalarType(common_dtype),
                          kernel,
                          op,
                          kernel_operator);

            [computeEncoder setComputePipelineState:binaryPSO];
            [computeEncoder setBuffer:inputBuffer  offset:self.storage_offset() * self.element_size() atIndex:0];
            [computeEncoder setBuffer:otherBuffer  offset:other.storage_offset() * other.element_size() atIndex:1];
            [computeEncoder setBuffer:outputBuffer offset:output.storage_offset() * output.element_size() atIndex:2];

            [computeEncoder setBytes:&src_sizes[0] length:sizeof(uint32_t) * kernel_size atIndex:3];
            [computeEncoder setBytes:&src_strides[0] length:sizeof(uint32_t) * kernel_size atIndex:4];

            [computeEncoder setBytes:&other_sizes[0] length:sizeof(uint32_t) * kernel_size atIndex:5];
            [computeEncoder setBytes:&other_strides[0] length:sizeof(uint32_t) * kernel_size atIndex:6];

            [computeEncoder setBytes:&output_sizes[0] length:sizeof(uint32_t) * kernel_size atIndex:7];
            [computeEncoder setBytes:&output_strides[0] length:sizeof(uint32_t) * kernel_size atIndex:8];

            uint32_t numThreads = self.numel();
            MTLSize gridSize = MTLSizeMake(numThreads, 1, 1);
            NSUInteger threadsPerThreadgroup_ = binaryPSO.maxTotalThreadsPerThreadgroup;
            if (threadsPerThreadgroup_ > numThreads) {
                threadsPerThreadgroup_ = numThreads;
            }

            MTLSize threadsPerThreadgroup = MTLSizeMake(threadsPerThreadgroup_, 1, 1);
            [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadsPerThreadgroup];
            mpsStream->commitAdaptive({self, other}, output, binaryPSO);
          }
        });
        return true;
      }
    }
  }


  TensorIterator iter;
  if (op == "lt" || op == "le" || op == "gt" || op == "ge" || op == "ne" || op == "logical_or" || op == "logical_and" || op == "eq") {
    iter = TensorIterator::comparison_op(const_cast<Tensor&>(output), self, other);
  } else {
    iter = TensorIterator::borrowing_binary_op(output, self, other);
  }

  dispatch_binary_kernel_mps_(iter, op, kernel_operator);
  return true;
}

bool getBinaryKernelOperator(const std::string& op_name, std::pair<std::string, std::string>& kernel_operator) {
  static bool macOS13_0_plus = is_macos_13_or_newer(MacOSVersion::MACOS_VER_13_0_PLUS);
  if (!macOS13_0_plus) {
    return false;
  }

  static std::unordered_map<std::string, std::pair<std::string, std::string>> opToKernelOperator = {
    {"multiplication",        {"mul", "*" }},
    {"div_out_mps:",          {"div", "/" }},
    {"add_out_mps:",          {"add", "+" }},
    {"sub_out_mps:",          {"sub", "-" }},

    // comparison ops
    {"lessThan",              {"lt",          "<" }},
    {"lessThanOrEqualTo",     {"le",          "<="}},
    {"greaterThan",           {"gt",          ">" }},
    {"greaterThanOrEqualTo",  {"ge",          ">="}},
    {"notEqual",              {"ne",          "!="}},
    {"logicalOR",             {"logical_or",  "||"}},
    {"logicalAND",            {"logical_and", "&&"}},
    {"equal",                 {"eq",          "=="}},
  };

  auto it = opToKernelOperator.find(op_name);
  if (it == opToKernelOperator.end()) {
    return false;
  }

  kernel_operator = it->second;
  return true;
}

bool dispatchNativeBinaryKernel(const Tensor& self,
                          const Tensor& other,
                          const Tensor& output,
                          const Scalar& alpha,
                          const std::string& op_name) {
  if (alpha.toFloat() == 1.0) {
    std::pair<std::string, std::string> kernel_operator;
    if (getBinaryKernelOperator(op_name, kernel_operator)) {
      return dispatch_binary_kernel_mps(self, other, output, kernel_operator.first, kernel_operator.second);
    }
  }

  return false;
}

static const char* METAL_BINARY = R"BINARY_METAL(

#include <metal_stdlib>
using namespace metal;

template<typename T>
kernel void fmax(constant void     * input_        [[buffer(0)]],
                  constant void     * other_        [[buffer(1)]],
                  device   void     * out_          [[buffer(2)]],
                  constant uint3    * offsets       [[buffer(3)]],
                  uint tid [[thread_position_in_grid]]) {
  device   T* out   = (device   T*)((device uint8_t*)out_ + offsets[tid].x);
  constant T* input = (constant T*)((constant uint8_t*)input_ + offsets[tid].y);
  constant T* other = (constant T*)((constant uint8_t*)other_ + offsets[tid].z);

  *out = fmax(*input, *other);
}

template<typename T>
kernel void fmin(constant void     * input_        [[buffer(0)]],
                  constant void     * other_        [[buffer(1)]],
                  device   void     * out_          [[buffer(2)]],
                  constant uint3    * offsets       [[buffer(3)]],
                  uint tid [[thread_position_in_grid]]) {
  device   T* out   = (device   T*)((device uint8_t*)out_ + offsets[tid].x);
  constant T* input = (constant T*)((constant uint8_t*)input_ + offsets[tid].y);
  constant T* other = (constant T*)((constant uint8_t*)other_ + offsets[tid].z);

  *out = fmin(*input, *other);
}

#define REGISTER_FMAX_OP(DTYPE)                       \
template                                               \
[[host_name("fmax_" #DTYPE)]]                         \
kernel void fmax<DTYPE>(                  \
  constant void     * input_        [[buffer(0)]],     \
  constant void     * other_        [[buffer(1)]],     \
  device   void     * out_          [[buffer(2)]],     \
  constant uint3    * offsets       [[buffer(3)]],     \
  uint tid [[thread_position_in_grid]]);

#define REGISTER_FMIN_OP(DTYPE)                       \
template                                               \
[[host_name("fmin_" #DTYPE)]]                         \
kernel void fmin<DTYPE>(                  \
  constant void     * input_        [[buffer(0)]],     \
  constant void     * other_        [[buffer(1)]],     \
  device   void     * out_          [[buffer(2)]],     \
  constant uint3    * offsets       [[buffer(3)]],     \
  uint tid [[thread_position_in_grid]]);

REGISTER_FMAX_OP(float);
REGISTER_FMAX_OP(half);
REGISTER_FMIN_OP(float);
REGISTER_FMIN_OP(half);

)BINARY_METAL";

using namespace mps;

static id<MTLLibrary> compileBinaryOpsLibrary(id<MTLDevice> device) {
  static id<MTLLibrary> binaryLibrary = nil;
  if (binaryLibrary) {
    return binaryLibrary;
  }

  NSError *error = nil;
  MTLCompileOptions *options = [[MTLCompileOptions new] autorelease];
  [options setLanguageVersion: MTLLanguageVersion2_3];
  binaryLibrary  = [device newLibraryWithSource:[NSString stringWithCString: METAL_BINARY encoding:NSASCIIStringEncoding]
                                       options:options
                                         error:&error];
  TORCH_CHECK(binaryLibrary, "Failed to create metal binary library, error: ", [[error description] UTF8String]);
  return binaryLibrary;
}

static id<MTLComputePipelineState> binaryPipelineState(id<MTLDevice> device, const std::string& kernel) {
  static std::unordered_map<std::string, id<MTLComputePipelineState>> psoCache;
  id<MTLComputePipelineState> pso = psoCache[kernel];
  if (pso) {
    return pso;
  }

  NSError* error = nil;
  id<MTLLibrary> binaryLib = compileBinaryOpsLibrary(device);
  id<MTLFunction> binaryFunc = [binaryLib newFunctionWithName:[NSString stringWithUTF8String:kernel.c_str()]];
  TORCH_CHECK(binaryFunc, "Failed to create function state object for: ", kernel);
  pso = [device newComputePipelineStateWithFunction:binaryFunc error:&error];
  TORCH_CHECK(pso, "Failed to created pipeline state object, error: ", [[error description] UTF8String]);

  psoCache[kernel] = pso;
  return pso;
}

void fmax_fmin_mps_impl(TensorIteratorBase& iter, const std::string max_min) {
  TORCH_CHECK(iter.common_dtype() != at::kDouble, "float64 is not supported on MPS");

  Tensor input = iter.input(0);
  Tensor other = iter.input(1);
  Tensor out = iter.output(0);
  id<MTLBuffer> inputBuffer  = getMTLBufferStorage(input);
  id<MTLBuffer> otherBuffer  = getMTLBufferStorage(other);
  id<MTLBuffer> outputBuffer = getMTLBufferStorage(out);
  id<MTLDevice> device = MPSDevice::getInstance()->device();
  MPSStream* mpsStream = getCurrentMPSStream();
  const uint32_t nDim = iter.ndim();
  constexpr uint32_t nOffsets = 3;
  const uint32_t numThreads = iter.numel();
  dispatch_sync(mpsStream->queue(), ^(){
    @autoreleasepool {
      NSError* error = nil;
      id<MTLComputeCommandEncoder> computeEncoder = mpsStream->commandEncoder();
      MTLSize gridSize = MTLSizeMake(numThreads, 1, 1);
      const IntArrayRef& iterShape = iter.shape();
      std::vector<uint32_t> iterShapeData(iterShape.size());
      std::vector<std::array<uint32_t, nOffsets>> strides(nDim);

      for (const auto i: c10::irange(iterShape.size())) {
        TORCH_CHECK(i <= UINT32_MAX);
        iterShapeData[i] = (uint32_t)(iterShape[i]);
      }

      for (const auto i: c10::irange(nDim)) {
        for (const auto offset: c10::irange(nOffsets)) {
            strides[i][offset] = iter.strides(offset)[i];
        }
      }

      id<MTLComputePipelineState> kernelDataOffsetsPSO = MPSDevice::getInstance()->metalIndexingFunction("kernel_index_offsets");
      id<MTLBuffer> kernelDataOffsets = [[device newBufferWithLength: numThreads * sizeof(simd_uint3)
                                                             options: 0] autorelease];
      TORCH_CHECK(kernelDataOffsetsPSO, "Failed to created pipeline state object, error: ", [[error description] UTF8String]);
      [computeEncoder setComputePipelineState:kernelDataOffsetsPSO];
      [computeEncoder setBytes:strides.data() length:sizeof(uint32_t) * nDim * nOffsets atIndex:0];
      [computeEncoder setBuffer:kernelDataOffsets offset:0 atIndex:1];
      [computeEncoder setBytes:iterShapeData.data() length:sizeof(uint32_t) * iterShape.size() atIndex:2];
      [computeEncoder setBytes:&nDim length:sizeof(uint32_t) atIndex:3];
      [computeEncoder setBytes:&nOffsets length:sizeof(uint32_t) atIndex:4];

      NSUInteger kernelOffsetsTGSize = kernelDataOffsetsPSO.maxTotalThreadsPerThreadgroup;
      if (kernelOffsetsTGSize > numThreads)
          kernelOffsetsTGSize = numThreads;

      MTLSize kernelOffsetsThreadGroupSize = MTLSizeMake(kernelOffsetsTGSize, 1, 1);
      [computeEncoder dispatchThreads: gridSize
                threadsPerThreadgroup: kernelOffsetsThreadGroupSize];

      const std::string kernel = "f" + max_min + "_" + scalarToMetalTypeString(out.scalar_type());
      id<MTLComputePipelineState> fmaxfminPSO = binaryPipelineState(device, kernel);

      // this function call is a no-op if MPS Profiler is not enabled
      getMPSProfiler().beginProfileKernel(fmaxfminPSO, kernel, {input, other});

      [computeEncoder setComputePipelineState:fmaxfminPSO];
      [computeEncoder setBuffer:inputBuffer  offset:input.storage_offset() * input.element_size() atIndex:0];
      [computeEncoder setBuffer:otherBuffer  offset:other.storage_offset() * other.element_size() atIndex:1];
      [computeEncoder setBuffer:outputBuffer offset:out.storage_offset() * out.element_size() atIndex:2];
      [computeEncoder setBuffer:kernelDataOffsets offset:0 atIndex:3];

      NSUInteger tgSize = fmaxfminPSO.maxTotalThreadsPerThreadgroup;
      if (tgSize > numThreads) {
          tgSize = numThreads;
      }

      MTLSize threadGroupSize = MTLSizeMake(tgSize, 1, 1);
      [computeEncoder dispatchThreads: gridSize
                threadsPerThreadgroup: threadGroupSize];
      mpsStream->commitAdaptive({input, other}, out, fmaxfminPSO);
    }
  });
}
} // namespace mps

void fmax_mps_kernel(TensorIteratorBase& iter) {
    if (isFloatingType(iter.common_dtype())) {
        mps::fmax_fmin_mps_impl(iter, "max");
    } else {
        at::maximum_out(const_cast<Tensor&>(iter.output()), iter.input(0), iter.input(1));
    }
}
void fmin_mps_kernel(TensorIteratorBase& iter) {
    if (isFloatingType(iter.common_dtype())) {
        mps::fmax_fmin_mps_impl(iter, "min");
    } else {
        at::minimum_out(const_cast<Tensor&>(iter.output()), iter.input(0), iter.input(1));
    }
}

REGISTER_DISPATCH(fmax_stub, &fmax_mps_kernel);
REGISTER_DISPATCH(fmin_stub, &fmin_mps_kernel);

} // namespace at::native
