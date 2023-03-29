//  Copyright © 2022 Apple Inc.

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/Utils.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>
#include <torch/library.h>
#include <c10/util/Optional.h>
#include <ATen/native/BinaryOps.h>
#include <fmt/format.h>
#include <ATen/mps/MPSAllocatorInterface.h>

namespace at::native {
namespace mps {

struct BinaryOpCachedGraph : public MPSCachedGraph
{
  BinaryOpCachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
  MPSGraphTensor *primaryTensor = nil, *secondaryTensor = nil;
  MPSGraphTensor *alphaTensor = nil, *outputTensor = nil;
};

typedef MPSGraphTensor* (^BinaryOpBlock)(BinaryOpCachedGraph*, MPSGraphTensor*, MPSGraphTensor*);
#define BinaryOpFn(graph, primary, secondary) MPSGraphTensor* (mps::BinaryOpCachedGraph* graph, MPSGraphTensor* primary, MPSGraphTensor* secondary)

enum class BinaryKernelType {
  RHS_Scalar,
  Tensor
};

static char* BINARY_OP_TEMPLATE_STRIDED_TENSOR = R"METAL_BINARY(
kernel void {3}_kernel_strided(uint tid                  [[thread_position_in_grid]],
                       constant void  * input_           [[buffer(0)]],
                       constant void  * other_           [[buffer(1)]],
                       device   void  * output_          [[buffer(2)]],
                       constant uint3 * offsets          [[buffer(3)]]) {{
  device   {0}* output = (device   {0}*)((device uint8_t*)output_  + offsets[tid].x);
  constant {1}* input  = (constant {1}*)((constant uint8_t*)input_ + offsets[tid].y);
  constant {2}* other  = (constant {2}*)((constant uint8_t*)other_ + offsets[tid].z);

  *output = *input {4} *other;
}}
)METAL_BINARY";

static  char* BINARY_OP_TEMPLATE_STRIDED_RHS_SCALAR = R"METAL_BINARY(
kernel void {3}_kernel_scalar_strided(uint tid           [[thread_position_in_grid]],
                       constant void  * input_           [[buffer(0)]],
                       constant {2}   & other            [[buffer(1)]],
                       device   void  * output_          [[buffer(2)]],
                       constant uint3 * offsets          [[buffer(3)]]) {{
  device   {0}* output = (device   {0}*)((device uint8_t*)output_  + offsets[tid].x);
  constant {1}* input  = (constant {1}*)((constant uint8_t*)input_ + offsets[tid].y);

  *output = *input {4} other;
}}
)METAL_BINARY";

const std::string& getMetalType(const c10::ScalarType& t) {
  static std::unordered_map<c10::ScalarType, std::string> scalar_to_metal_type = {
    {c10::ScalarType::Float, "float"},
    {c10::ScalarType::Half,  "half"},
    {c10::ScalarType::Long,  "long"},
    {c10::ScalarType::Int,   "int"},
    {c10::ScalarType::Short, "short"},
    {c10::ScalarType::Char,  "char"},
    {c10::ScalarType::Byte,  "uchar"},
    {c10::ScalarType::Bool,  "bool"},
  };

  auto it = scalar_to_metal_type.find(t);
  TORCH_CHECK(it != scalar_to_metal_type.end(), "Unsupported type ", t);
  return it->second;
}

const std::string& getMetalType(const at::Tensor& t) {
  return getMetalType(t.scalar_type());
}

const std::string& getMetalType(const c10::Scalar& s) {
  return getMetalType(s.type());
}

static id<MTLLibrary> compileBinaryOpsLibrary(id<MTLDevice> device,
                                               const std::string& t1,
                                               const std::string& t2,
                                               const std::string& t3,
                                               const std::string& op,
                                               const std::string& kernel_operator,
                                               BinaryKernelType binaryKernelType) {
  auto key = op + t1 + t2 + t3 + std::to_string(int(binaryKernelType));
  // std::cout << key << std::endl;
  // std::cout << "LIB: " << key << std::endl;
  static std::unordered_map<std::string, id<MTLLibrary>> libMap;
  auto it = libMap.find(key);
  if (it != libMap.end()) {
    return it->second;
  }
  NSError *error = nil;
  MTLCompileOptions *options = [[MTLCompileOptions new] autorelease];
  [options setLanguageVersion: MTLLanguageVersion3_0];
  char *str = nil;
  switch (binaryKernelType){
    case BinaryKernelType::RHS_Scalar:
      str = BINARY_OP_TEMPLATE_STRIDED_RHS_SCALAR;
      break;
    case BinaryKernelType::Tensor:
      str = BINARY_OP_TEMPLATE_STRIDED_TENSOR;
      break;
    default:
      TORCH_CHECK(false);
      assert(0);
  }

  auto rc  = [device newLibraryWithSource:[NSString stringWithUTF8String:fmt::format(str, t1, t2, t3, op, kernel_operator).c_str()]
                                  options:options
                                    error:&error];
 TORCH_CHECK(rc != nil && error == nil, "Failed to compile library: ", [[error localizedDescription] UTF8String]);
 libMap[key] = rc;
 return rc;
}


static id<MTLComputePipelineState> getBinaryPSO(id<MTLDevice> device,
                                                const std::string& t1, // output
                                                const std::string& t2, // input
                                                const std::string& t3, // other
                                                const std::string& fname,
                                                const std::string& op,
                                                const std::string& kernel_operator,
                                                BinaryKernelType binaryKernelType) {
  auto key = t1 + t2 + t3 + fname;
  // std::cout << "FUNC KEY: " << key << std::endl;
  // std::cout << "PSO: " << key << std::endl;
  static std::unordered_map<std::string, id<MTLComputePipelineState>> cplMap;
  auto it = cplMap.find(key);
  if (it != cplMap.end()) {
     return it->second;
  }
  NSError *error = nil;
  auto library = compileBinaryOpsLibrary(device, t1, t2, t3, op, kernel_operator, binaryKernelType);
  id<MTLFunction> func = [library newFunctionWithName:[NSString stringWithUTF8String:fname.c_str()]];
  TORCH_CHECK(func != nil, "Can't get function ", fname);
  auto rc = [device newComputePipelineStateWithFunction:func error:&error];
  TORCH_CHECK(rc != nil && error == nil, "Failed to construct pipeline state: ", [[error localizedDescription] UTF8String]);
  cplMap[key]  = rc;
  return rc;
}

void binary_kernel_mps_(TensorIteratorBase& iter, const std::string& op, const std::string& kernel_operator) {
  Tensor inputTensor;
  Tensor otherTensor;
  BinaryKernelType type = BinaryKernelType::Tensor;

  int scalar_pos = 0;
  if (iter.is_scalar(1) || iter.is_scalar(2)) {
    type = BinaryKernelType::RHS_Scalar;
    scalar_pos = iter.is_scalar(1) ? 1 : 2;
    int tensor_pos = scalar_pos == 1 ? 2 : 1;
    inputTensor = iter.tensor(tensor_pos);
    otherTensor = iter.tensor(scalar_pos);
  } else {
    inputTensor = iter.tensor(1);
    otherTensor = iter.tensor(2);
  }

  if (inputTensor.numel() == 0 || otherTensor.numel() == 0) {
    return;
  }

  if (inputTensor.scalar_type() == kDouble) {
    inputTensor = inputTensor.to(iter.common_dtype());
  }
  if (otherTensor.scalar_type() == kDouble) {
    otherTensor = otherTensor.to(iter.common_dtype());
  }

  const Tensor& outputTensor = iter.tensor(0);

  // std::cout << iter.dtype() << std::endl;
  // std::cout << "Input dtype: " << inputTensor.dtype() << std::endl;
  // std::cout << "Other dtype: " << otherTensor.dtype() << std::endl;
  // std::cout << "output dtype: " << outputTensor.dtype() << std::endl;
  // std::cout << "COMMON DTYPE:" << iter.common_dtype() << std::endl;
  // std::cout << "ITER SCALAR(0): "<< iter.is_scalar(0) << " " << iter.is_cpu_scalar(0) << std::endl;
  // std::cout << "ITER SCALAR(1): "<< iter.is_scalar(1) << " " << iter.is_cpu_scalar(1) << std::endl;
  // std::cout << "ITER SCALAR(2): "<< iter.is_scalar(2) << " " << iter.is_cpu_scalar(2) << std::endl;
  // std::cout << "Input Sizes:" << inputTensor.sizes() << " numel: " << inputTensor.numel() << " contg: " << inputTensor.is_contiguous() << std::endl;
  // std::cout << "Other Sizes:" << otherTensor.sizes() << " numel: " << otherTensor.numel() << " contg: " << otherTensor.is_contiguous() << std::endl;
  // std::cout << "Output Sizes:" << outputTensor.sizes() << " numel: " << outputTensor.numel() << " contg: " << outputTensor.is_contiguous() << std::endl;

  MPSStream* mpsStream = getCurrentMPSStream();
  id<MTLDevice> device = MPSDevice::getInstance()->device();

  id<MTLBuffer> inputBuffer  = mps::getMTLBufferStorage(inputTensor);
  id<MTLBuffer> otherBuffer  = mps::getMTLBufferStorage(otherTensor);
  id<MTLBuffer> outputBuffer = mps::getMTLBufferStorage(outputTensor);
  mps::MPSScalar scalar;
  if (scalar_pos) {
    scalar = mps::getMPSScalar(otherTensor.item(), otherTensor.scalar_type());
    otherBuffer = (id<MTLBuffer>)getIMPSAllocator()->allocScalarBufferWithValue(&scalar.value, scalar.size).get();
  }
  // if (iter.is_cpu_scalar(1)) {
  //   scalar = mps::getMPSScalar(inputTensor.item(), inputTensor.scalar_type());
  //   inputBuffer = (id<MTLBuffer>)getIMPSAllocator()->allocScalarBufferWithValue(&scalar.value, scalar.size).get();
  // }
  // if (iter.is_cpu_scalar(2)) {
  //   scalar = mps::getMPSScalar(otherTensor.item(), otherTensor.scalar_type());
  //   otherBuffer = (id<MTLBuffer>)getIMPSAllocator()->allocScalarBufferWithValue(&scalar.value, scalar.size).get();
  // }

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

      id<MTLBuffer> kernelDataOffsets;
      if (!iter.is_contiguous()) {
        id<MTLComputePipelineState> kernelDataOffsetsPSO = MPSDevice::getInstance()->metalIndexingFunction("kernel_index_offsets");
        kernelDataOffsets = (id<MTLBuffer>)getIMPSAllocator()->allocate(numThreads * sizeof(simd_uint3)).get();
        TORCH_CHECK(kernelDataOffsetsPSO, "Failed to created pipeline state object, error: ", [[error description] UTF8String]);
        [computeEncoder setComputePipelineState:kernelDataOffsetsPSO];
        [computeEncoder setBytes:strides.data() length:sizeof(uint32_t) * nDim * nOffsets atIndex:0];
        [computeEncoder setBuffer:kernelDataOffsets offset:0 atIndex:1];
        [computeEncoder setBytes:iterShapeData.data() length:sizeof(uint32_t) * iterShape.size() atIndex:2];
        [computeEncoder setBytes:&nDim length:sizeof(uint32_t) atIndex:3];
        [computeEncoder setBytes:&nOffsets length:sizeof(uint32_t) atIndex:4];

        NSUInteger kernelOffsetsTGSize = kernelDataOffsetsPSO.maxTotalThreadsPerThreadgroup;
        if (kernelOffsetsTGSize > numThreads) {
            kernelOffsetsTGSize = numThreads;
        }

        MTLSize kernelOffsetsThreadGroupSize = MTLSizeMake(kernelOffsetsTGSize, 1, 1);
        [computeEncoder dispatchThreads: gridSize
                  threadsPerThreadgroup: kernelOffsetsThreadGroupSize];
      }

      std::string kernel = op;
      kernel += "_kernel";
      if (scalar_pos) {
        kernel += "_scalar";
      }
      if (!iter.is_contiguous()) {
        kernel += "_strided";
      }

      id<MTLComputePipelineState> binaryPSO = mps::getBinaryPSO(device,
                                                          mps::getMetalType(outputTensor),
                                                          mps::getMetalType(inputTensor),
                                                          mps::getMetalType(otherTensor),
                                                          kernel,
                                                          op,
                                                          kernel_operator,
                                                          type);
      [computeEncoder setComputePipelineState:binaryPSO];

      [computeEncoder setBuffer:inputBuffer  offset:inputTensor.storage_offset() * inputTensor.element_size() atIndex:0];
      [computeEncoder setBuffer:otherBuffer  offset:otherTensor.storage_offset() * otherTensor.element_size() atIndex:1];
      [computeEncoder setBuffer:outputBuffer offset:outputTensor.storage_offset() * outputTensor.element_size() atIndex:2];
      if (!iter.is_contiguous()) {
        [computeEncoder setBuffer:kernelDataOffsets offset:0 atIndex:3];
      }

      NSUInteger tgSize = binaryPSO.maxTotalThreadsPerThreadgroup;
      if (tgSize > numThreads) {
          tgSize = numThreads;
      }

      MTLSize threadGroupSize = MTLSizeMake(tgSize, 1, 1);
      [computeEncoder dispatchThreads: gridSize
                threadsPerThreadgroup: threadGroupSize];
    }
  });
}

static
void binary_kernel_mps(const Tensor& self, const Tensor& other, const Tensor& output, std::string op) {
  auto iter = TensorIterator::borrowing_binary_op(output, self, other);
  std::string kernel_operator;
  if (op == "mul") {
    kernel_operator = "*";
  } else if (op == "add") {
    kernel_operator = "+";
  } else {
    TORCH_CHECK(false, "Unsupported op");
  }
  binary_kernel_mps_(iter, op, kernel_operator);
}

// void add_kernel_mps(TensorIteratorBase& iter) {
//   // auto iter = TensorIterator::borrowing_binary_op(output, self, other);
//   // add_kernel_mps(iter);
//   mul_kernel_mps2(iter, "add");
// }

// alpha is always 1.0 except when this function is called from add_sub_template()
void binaryOpTensor(const Tensor& self, const Tensor& other, const Scalar& alpha,
                    const Tensor& output_, std::string op_name, BinaryOpBlock binaryBlock)
{
  TORCH_CHECK(!(!is_macos_13_or_newer() && self.scalar_type() == ScalarType::Byte ),
              "MPS support binary op with uint8 natively starting from macOS 13.0");
  TORCH_CHECK(!(op_name == "power" && !is_macos_13_or_newer(MacOSVersion::MACOS_VER_13_2_PLUS) &&
              (self.scalar_type() == ScalarType::Long ||
              (other.scalar_type() == ScalarType::Long && (self.scalar_type() != ScalarType::Half && self.scalar_type() != ScalarType::Float)))),
              "MPS: ", op_name, " op with int64 input is supported natively starting from macOS 13.2");
  MPSStream* mpsStream = getCurrentMPSStream();

  const bool is_self_scalar = self.dim() == 0;
  const bool is_other_scalar = other.dim() == 0;
  bool disableTypeInference = false;

  auto new_size = at::infer_size(self.sizes(), other.sizes());
  if (!output_.sizes().equals(new_size)) {
      output_.resize_(new_size);
  }

  // it's possible to receive empty tensors here
  if (self.numel() == 0 || other.numel() == 0) {
    return;
  }

  // std::cout << "Op name: " << op_name << std::endl;
  if (op_name == "multiplication" && (!self.is_contiguous() || !other.is_contiguous() || !output_.is_contiguous())) {
    // std::cout << "Using kernel\n";
    binary_kernel_mps(self, other, output_, "mul");
    return;
  }

  if (self.dim() == 1 || other.dim() == 1 || self.dim() >= 5 || other.dim() >= 5) {
    disableTypeInference = true;
  }

  Tensor output;
  bool needsCopyToOutput = false;

  // determine if this is an in-place operation
  if (self.is_alias_of(output_) || other.is_alias_of(output_)) {
    if (output_.storage_offset() || !output_.is_contiguous()) {
      output = at::native::empty_mps(output_.sizes(), output_.scalar_type(), c10::nullopt, kMPS);
      needsCopyToOutput = true;
    }
  } else if (!output_.is_contiguous()) {
    output_.unsafeGetTensorImpl()->empty_tensor_restride(MemoryFormat::Contiguous);
  }

  auto inputDataType = self.scalar_type();
  auto otherDataType = other.scalar_type();
  auto outputDataType = output_.scalar_type();
  if (!is_macos_13_or_newer()) {
    // workaround for signed vs. unsigned comparison issue in MacOS 12
    if (outputDataType == kBool && (inputDataType == kByte || otherDataType == kByte)) {
      inputDataType = otherDataType = kByte;
    } else {
      if (inputDataType == kBool || inputDataType == kByte) {
        inputDataType = kChar;
      }
      if (otherDataType == kBool || otherDataType == kByte) {
        otherDataType = kChar;
      }
    }
  }

  bool flattenTensors = false;
  MPSShape* selfShape = getMPSShape(self);
  MPSShape* otherShape = getMPSShape(other);
  MPSShape* outputShape = nil;
  std::vector<Tensor> broadcastTensors;
  if (self.dim() >= 5 && other.dim() >= 5) {
     flattenTensors = true;
     broadcastTensors = expand_outplace({self, other});
     selfShape = getMPSShape(broadcastTensors[0].numel());
     otherShape = getMPSShape(broadcastTensors[1].numel());
     outputShape = getMPSShape(output_.numel());
  }

  MPSGraphCache* cache_ = MPSGraphCache::getInstance();
  @autoreleasepool {
    string key = op_name + getTensorsStringKey({
      flattenTensors ? broadcastTensors[0] : self,
      flattenTensors ? broadcastTensors[1] : other,
      output_},  /*short_dtype=*/false,
      /*disable_type_inference=*/disableTypeInference);
    BinaryOpCachedGraph* cachedGraph = static_cast<BinaryOpCachedGraph *>(cache_->LookUp(key));

    if(!cachedGraph) {
      MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ MPSCachedGraph* () {
        BinaryOpCachedGraph *newCachedGraph = nil;
        @autoreleasepool {
          MPSGraph* mpsGraph = make_mps_graph();
          newCachedGraph = new BinaryOpCachedGraph(mpsGraph);
          if (disableTypeInference) {
            newCachedGraph->primaryTensor   = mpsGraphUnrankedPlaceHolder(mpsGraph, getMPSScalarType(inputDataType));
            newCachedGraph->secondaryTensor = mpsGraphUnrankedPlaceHolder(mpsGraph, getMPSScalarType(otherDataType));
          } else {
            newCachedGraph->primaryTensor   = mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(inputDataType), selfShape);
            newCachedGraph->secondaryTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(otherDataType), otherShape);
          }

          MPSGraphTensor* primaryCastTensor   = newCachedGraph->primaryTensor;
          MPSGraphTensor* secondaryCastTensor = newCachedGraph->secondaryTensor;

          // this type inference is only required at the time of graph creation
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
          if (inputDataType != common_dtype) {
            primaryCastTensor = castMPSTensor(mpsGraph, newCachedGraph->primaryTensor, common_dtype);
          }
          if (otherDataType != common_dtype) {
            secondaryCastTensor = castMPSTensor(mpsGraph, newCachedGraph->secondaryTensor, common_dtype);
          }
          newCachedGraph->outputTensor = binaryBlock(newCachedGraph, primaryCastTensor, secondaryCastTensor);
          // Cast output tensor to an expected type if needed, which addresses discrepancy when int64 scalar is added to int32 tensor
          // Output tensor should have been promoted but it remains an int32 tensor
          if (outputDataType != common_dtype ||
             [newCachedGraph->outputTensor dataType] != getMPSDataType(outputDataType)) {
            newCachedGraph->outputTensor = castMPSTensor(mpsGraph, newCachedGraph->outputTensor, outputDataType);
          }
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<BinaryOpCachedGraph *>(tmpCachedGraph);
    }

    NSMutableDictionary *feeds = [[NSMutableDictionary new] autorelease];
    Placeholder selfPlaceholder;
    Placeholder otherPlaceholder;
    MPSScalar self_scalar;
    MPSScalar other_scalar;
    MPSScalar alpha_scalar;

    if (is_self_scalar && !self.is_mps()) {
      self_scalar = getMPSScalar(self.item(), inputDataType);
      feeds[cachedGraph->primaryTensor] = getMPSGraphTensorFromScalar(mpsStream, self_scalar);
    } else {
      selfPlaceholder = Placeholder(cachedGraph->primaryTensor, flattenTensors ? broadcastTensors[0] : self, flattenTensors ? selfShape : nil,
                                    /*gatherTensorData=*/true, getMPSScalarType(inputDataType));
      feeds[selfPlaceholder.getMPSGraphTensor()] = selfPlaceholder.getMPSGraphTensorData();
    }
    if (is_other_scalar && !other.is_mps()) {
      other_scalar = getMPSScalar(other.item(), otherDataType);
      feeds[cachedGraph->secondaryTensor] = getMPSGraphTensorFromScalar(mpsStream, other_scalar);
    } else {
      otherPlaceholder = Placeholder(cachedGraph->secondaryTensor, flattenTensors ? broadcastTensors[1] : other,  flattenTensors ? otherShape : nil,
                                     /*gatherTensorData=*/true, getMPSScalarType(otherDataType));
      feeds[otherPlaceholder.getMPSGraphTensor()] = otherPlaceholder.getMPSGraphTensorData();
    }

    // 'cachedGraph->alphaTensor' is not nil only if add_sub_template() was called with an alpha value != 1.0
    if (cachedGraph->alphaTensor) {
      alpha_scalar = getMPSScalar(alpha, other.scalar_type());
      feeds[cachedGraph->alphaTensor] = getMPSGraphTensorFromScalar(mpsStream, alpha_scalar);
    }

    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor, needsCopyToOutput ? output : output_, outputShape);
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    runMPSGraph(mpsStream, cachedGraph->graph(), feeds, results, disableTypeInference);

    if (needsCopyToOutput) {
      output_.copy_(output);
    }
  }
}

void binaryOpScalar(const Tensor& self, const Scalar& other, const Scalar& alpha,
                    const Tensor& output, std::string op_name, BinaryOpBlock binaryBlock)
{
  binaryOpTensor(self, wrapped_scalar_tensor(other), alpha, output, op_name, binaryBlock);
}

void div_mode_template(const Tensor& self, const Tensor& other,
                       c10::optional<c10::string_view> rounding_mode,
                       const Tensor& output, const string op_name)
{
  if(rounding_mode.has_value() && *rounding_mode == "trunc"){
    TORCH_CHECK(self.scalar_type() != ScalarType::Half,
                "MPS: does not support trunc_divide op with float16 input");
  }
  BinaryOpBlock div_mode_op_block = ^BinaryOpFn(cachedGraph, primaryCastTensor, secondaryCastTensor) {
    MPSGraph* mpsGraph = cachedGraph->graph();
    bool isFloatInput = ([primaryCastTensor dataType] & MPSDataTypeFloatBit) != 0;
    if(!isFloatInput && rounding_mode.has_value() && (*rounding_mode == "floor" || *rounding_mode == "trunc")) {
      primaryCastTensor = [mpsGraph castTensor:primaryCastTensor
                                        toType:MPSDataTypeFloat32
                                          name:@"primaryCastTensor"];
      secondaryCastTensor = [mpsGraph castTensor:secondaryCastTensor
                                          toType:MPSDataTypeFloat32
                                            name:@"secondaryCastTensor"];
    }
    MPSGraphTensor* divTensor =  [mpsGraph divisionWithPrimaryTensor:primaryCastTensor
                                                     secondaryTensor:secondaryCastTensor
                                                                name:nil];
    // Rounding is a no-op for integral types, and also a reasonable workaround
    // For MPSGraph bug on Apple Silicon, that throws `Function floorOp_i64 was not found in the library`
    // See https://github.com/pytorch/pytorch/issues/84995
    bool isFloatOutput = ([divTensor dataType] & MPSDataTypeFloatBit) != 0;
    if (!rounding_mode.has_value() || !isFloatOutput) {
      return divTensor;
    } else if (*rounding_mode == "trunc") {
      auto truncTensor =  trunc_tensor(mpsGraph, divTensor);
      if (op_name == "fmod_mps_out") {
        auto mulTensor = [mpsGraph multiplicationWithPrimaryTensor:truncTensor
                                                   secondaryTensor:secondaryCastTensor
                                                              name:nil];
        return [mpsGraph subtractionWithPrimaryTensor:primaryCastTensor
                                      secondaryTensor:mulTensor
                                                 name:nil];
      }
      return truncTensor;
    } else if (*rounding_mode == "floor") {
      MPSGraphTensor* floorTensor = [mpsGraph floorWithTensor:divTensor name:nil];
      if (op_name == "remainder_out_mps") {
        auto mulTensor = [mpsGraph multiplicationWithPrimaryTensor:floorTensor
                                                   secondaryTensor:secondaryCastTensor
                                                              name:nil];
        return [mpsGraph subtractionWithPrimaryTensor:primaryCastTensor
                                      secondaryTensor:mulTensor
                                                 name:nil];
      }
      return floorTensor;
    }
    assert(0 && "Invalid rounding mode\n");
    return nullptr;
  };
  binaryOpTensor(self, other, Scalar(1.0), output, op_name + "_mps:" + (rounding_mode.has_value() ? c10::str(*rounding_mode) : ""), div_mode_op_block);
}

void add_sub_template(const Tensor& self, const Tensor& other, const Scalar& alpha, const Tensor& output, std::string op_name)
{
  if (alpha.toDouble() == 0.0) {
    const_cast<Tensor&>(output) = self.clone();
    return;
  }

  const bool alpha_has_value = alpha.toDouble() != 1.0;
  if (alpha_has_value) {
    auto commonDtype = at::result_type(self, other);
    at::native::alpha_check(commonDtype, alpha);
  }

  BinaryOpBlock add_sub_op_block = ^BinaryOpFn(cachedGraph, primaryCastTensor, secondaryCastTensor) {
    MPSGraph* mpsGraph = cachedGraph->graph();
    MPSGraphTensor* secondaryTensor = secondaryCastTensor;

    // if alpha is 1.0, then we don't bother adding another multiply to graph
    if (alpha_has_value) {
      cachedGraph->alphaTensor = mpsGraphRankedPlaceHolder(mpsGraph, getMPSScalarType(other.scalar_type()), @[@1]);
      secondaryTensor = [mpsGraph multiplicationWithPrimaryTensor:secondaryCastTensor
                                                  secondaryTensor:cachedGraph->alphaTensor
                                                             name:nil];
    }
    if (op_name == "add")
      return [mpsGraph additionWithPrimaryTensor:primaryCastTensor
                                 secondaryTensor:secondaryTensor
                                            name:nil];
    else
      return [mpsGraph subtractionWithPrimaryTensor:primaryCastTensor
                                    secondaryTensor:secondaryTensor
                                               name:nil];
  };
  // add alpha's type to the key only if multiply was added to graph
  binaryOpTensor(self, other, alpha, output, op_name + "_out_mps:" + (alpha_has_value ? getMPSTypeString(alpha.type()) : ""), add_sub_op_block);
}

} // namespace mps

#define CREATE_MPS_BINARY_COMPARISON_OP_FUNC(func_out, func_stub, other_type)                                             \
Tensor& func_out (const Tensor& self, const other_type& other, Tensor& output) {                                          \
  mps::binaryOp##other_type(self, other, Scalar(1.0), output, #func_stub,                                                 \
    ^BinaryOpFn(cachedGraph, primaryCastTensor, secondaryCastTensor) {                                                    \
      MPSGraph* mpsGraph = cachedGraph->graph();                                                                          \
      return [mpsGraph func_stub##WithPrimaryTensor:mps::castMPSTensor(mpsGraph, primaryCastTensor, ScalarType::Bool)     \
                                    secondaryTensor:mps::castMPSTensor(mpsGraph, secondaryCastTensor, ScalarType::Bool)   \
                                               name:nil]; });                                                             \
  return output;                                                                                                          \
}

#define CREATE_MPS_STRUCTURED_BINARY_OP_FUNC(func_out, func_stub, other_type)                   \
TORCH_IMPL_FUNC(func_out) (const Tensor& self, const other_type& other, const Tensor& output) { \
  TORCH_CHECK(!(self.scalar_type() == ScalarType::Long &&                                       \
               std::string(#func_stub) == "atan2"),                                             \
               "MPS does not support ", #func_stub, " op with int64 input")                     \
  mps::binaryOp##other_type(self, other, Scalar(1.0), output, #func_stub,                       \
    ^BinaryOpFn(cachedGraph, primaryCastTensor, secondaryCastTensor) {                          \
      MPSGraph* mpsGraph = cachedGraph->graph();                                                \
      return [mpsGraph func_stub##WithPrimaryTensor:primaryCastTensor                           \
                                    secondaryTensor:secondaryCastTensor                         \
                                               name:nil]; });                                   \
}

// output of Boolean Ops will be cast to "MPSDataTypeBool" at the end of binaryOpTensor()
#define CREATE_MPS_STRUCTURED_BOOLEAN_OP_FUNC(func_out, func_stub, other_type)                  \
TORCH_IMPL_FUNC(func_out) (const Tensor& self, const other_type& other, const Tensor& output) { \
  mps::binaryOp##other_type(self, other, Scalar(1.0), output, #func_stub,                       \
    ^BinaryOpFn(cachedGraph, primaryCastTensor, secondaryCastTensor) {                          \
      MPSGraph* mpsGraph = cachedGraph->graph();                                                \
      return [mpsGraph func_stub##WithPrimaryTensor:primaryCastTensor                           \
                                    secondaryTensor:secondaryCastTensor                         \
                                               name:nil]; });                                   \
}

// Boolean Binary Ops
CREATE_MPS_STRUCTURED_BOOLEAN_OP_FUNC(eq_scalar_out_mps, equal, Scalar);
CREATE_MPS_STRUCTURED_BOOLEAN_OP_FUNC(eq_tensor_out_mps, equal, Tensor);
CREATE_MPS_STRUCTURED_BOOLEAN_OP_FUNC(ne_scalar_out_mps, notEqual, Scalar);
CREATE_MPS_STRUCTURED_BOOLEAN_OP_FUNC(ne_tensor_out_mps, notEqual, Tensor);
CREATE_MPS_STRUCTURED_BOOLEAN_OP_FUNC(le_scalar_out_mps, lessThanOrEqualTo, Scalar);
CREATE_MPS_STRUCTURED_BOOLEAN_OP_FUNC(le_tensor_out_mps, lessThanOrEqualTo, Tensor);
CREATE_MPS_STRUCTURED_BOOLEAN_OP_FUNC(lt_scalar_out_mps, lessThan, Scalar);
CREATE_MPS_STRUCTURED_BOOLEAN_OP_FUNC(lt_tensor_out_mps, lessThan, Tensor);
CREATE_MPS_STRUCTURED_BOOLEAN_OP_FUNC(ge_scalar_out_mps, greaterThanOrEqualTo, Scalar);
CREATE_MPS_STRUCTURED_BOOLEAN_OP_FUNC(ge_tensor_out_mps, greaterThanOrEqualTo, Tensor);
CREATE_MPS_STRUCTURED_BOOLEAN_OP_FUNC(gt_scalar_out_mps, greaterThan, Scalar);
CREATE_MPS_STRUCTURED_BOOLEAN_OP_FUNC(gt_tensor_out_mps, greaterThan, Tensor);

// Arithmetic Binary Ops
CREATE_MPS_STRUCTURED_BINARY_OP_FUNC(minimum_out_mps, minimum, Tensor);
CREATE_MPS_STRUCTURED_BINARY_OP_FUNC(maximum_out_mps, maximum, Tensor);
CREATE_MPS_STRUCTURED_BINARY_OP_FUNC(mul_out_mps, multiplication, Tensor);
CREATE_MPS_STRUCTURED_BINARY_OP_FUNC(pow_tensor_scalar_out_mps, power, Scalar);
CREATE_MPS_STRUCTURED_BINARY_OP_FUNC(pow_tensor_tensor_out_mps, power, Tensor);
CREATE_MPS_STRUCTURED_BINARY_OP_FUNC(atan2_mps_out, atan2, Tensor);

CREATE_MPS_BINARY_COMPARISON_OP_FUNC(logical_and_out_mps, logicalAND, Tensor);
CREATE_MPS_BINARY_COMPARISON_OP_FUNC(logical_or_out_mps, logicalOR, Tensor);
CREATE_MPS_BINARY_COMPARISON_OP_FUNC(logical_xor_out_mps, logicalXOR, Tensor);


TORCH_IMPL_FUNC(div_out_mode_mps) (const Tensor& self, const Tensor& other, c10::optional<c10::string_view> rounding_mode, const Tensor& output) {
  mps::div_mode_template(self, other, rounding_mode, output, "div_mode_out");
}

TORCH_IMPL_FUNC(div_out_mps) (const Tensor& self, const Tensor& other, const Tensor& output) {
  mps::div_mode_template(self, other, c10::nullopt, output, "div_out");
}

TORCH_IMPL_FUNC(add_out_mps) (const Tensor& self, const Tensor& other, const Scalar& alpha, const Tensor& output) {
  mps::add_sub_template(self, other, alpha, output, "add");
}

TORCH_IMPL_FUNC(sub_out_mps) (const Tensor& self, const Tensor& other, const Scalar& alpha, const Tensor& output) {
  mps::add_sub_template(self, other, alpha, output, "sub");
}

Tensor& floor_divide_out_mps(const Tensor& self, const Tensor& other, Tensor& result) {
  mps::div_mode_template(self, other, "floor", result, "floor_divide_out");
  return result;
}

Tensor floor_divide_mps(const Tensor& self, const Tensor& other) {
  Tensor output = at::empty_like(self);
  mps::div_mode_template(self, other, "floor", output, "floor_divide");
  return output;
}

Tensor& floor_divide_mps_(Tensor& self, const Tensor& other) {
  return floor_divide_out_mps(self, other, self);
}

TORCH_IMPL_FUNC(remainder_out_mps) (const Tensor& self, const Tensor& other, const Tensor& output) {
  mps::div_mode_template(self, other, "floor", output, "remainder_out_mps");
}

TORCH_IMPL_FUNC(fmod_mps_out) (const Tensor& self, const Tensor& other, const Tensor& output) {
  mps::div_mode_template(self, other, "trunc", output, "fmod_mps_out");
}

TORCH_IMPL_FUNC(hypot_out_mps) (const Tensor& self, const Tensor& other, const Tensor& output)
{
  mps::BinaryOpBlock hypot_op_block = ^BinaryOpFn(cachedGraph, primaryCastTensor, secondaryCastTensor) {
    MPSGraph* mpsGraph = cachedGraph->graph();
    MPSGraphTensor* twoTensor = [mpsGraph constantWithScalar:2.0
                                                       shape:@[@1]
                                                    dataType:primaryCastTensor.dataType];
    MPSGraphTensor* sumTensor = [mpsGraph additionWithPrimaryTensor:[mpsGraph powerWithPrimaryTensor:primaryCastTensor
                                                                                     secondaryTensor:twoTensor
                                                                                                name:nil]
                                                    secondaryTensor:[mpsGraph powerWithPrimaryTensor:secondaryCastTensor
                                                                                     secondaryTensor:twoTensor
                                                                                                name:nil]
                                                               name:nil];
    return [mpsGraph squareRootWithTensor:sumTensor name:nil];
  };
  mps::binaryOpTensor(self, other, Scalar(1.0), output, "hypot_out_mps", hypot_op_block);
}

TORCH_IMPL_FUNC(logaddexp_out_mps) (const Tensor& self, const Tensor& other, const Tensor& output)
{
  mps::BinaryOpBlock logaddexp_op_block = ^BinaryOpFn(cachedGraph, primaryCastTensor, secondaryCastTensor) {
    MPSGraph* mpsGraph = cachedGraph->graph();
    MPSGraphTensor* sumTensor = [mpsGraph additionWithPrimaryTensor:[mpsGraph exponentWithTensor:primaryCastTensor name:nil]
                                                    secondaryTensor:[mpsGraph exponentWithTensor:secondaryCastTensor name:nil]
                                                               name:nil];
    return [mpsGraph logarithmWithTensor:sumTensor name:nil];
  };
  mps::binaryOpTensor(self, other, Scalar(1.0), output, "logaddexp_out_mps", logaddexp_op_block);
}

TORCH_IMPL_FUNC(logaddexp2_out_mps) (const Tensor& self, const Tensor& other, const Tensor& output)
{
 mps::BinaryOpBlock logaddexp2_op_block = ^BinaryOpFn(cachedGraph, primaryCastTensor, secondaryCastTensor) {
    MPSGraph* mpsGraph = cachedGraph->graph();
    MPSGraphTensor* sumTensor = [mpsGraph additionWithPrimaryTensor:[mpsGraph exponentBase2WithTensor:primaryCastTensor name:nil]
                                                    secondaryTensor:[mpsGraph exponentBase2WithTensor:secondaryCastTensor name:nil]
                                                               name:nil];
    return [mpsGraph logarithmBase2WithTensor:sumTensor name:nil];
  };
  mps::binaryOpTensor(self, other, Scalar(1.0), output, "logaddexp2_out_mps", logaddexp2_op_block);
}

TORCH_IMPL_FUNC(xlogy_out_mps) (const Tensor& self, const Tensor& other, const Tensor& output) {
  mps::BinaryOpBlock xlogy_op_block = ^BinaryOpFn(cachedGraph, primaryCastTensor, secondaryCastTensor) {
    MPSGraph* mpsGraph = cachedGraph->graph();
    MPSGraphTensor* zeroTensor = [mpsGraph constantWithScalar:0.0
                                                        shape:@[@1]
                                                     dataType:primaryCastTensor.dataType];
    MPSGraphTensor* yIsNaNPredicateTensor = [mpsGraph isNaNWithTensor:secondaryCastTensor
                                                        name:nil];
    MPSGraphTensor* logyTensor = [mpsGraph logarithmWithTensor:secondaryCastTensor
                                                          name:nil];
    MPSGraphTensor* xlogyTensor = [mpsGraph multiplicationWithPrimaryTensor:primaryCastTensor
                                                            secondaryTensor:logyTensor
                                                                       name:nil];
    MPSGraphTensor* xEqualZeroPredicateTensor = [mpsGraph equalWithPrimaryTensor:primaryCastTensor
                                                        secondaryTensor:zeroTensor
                                                                   name:nil];
    MPSGraphTensor* outputTensor = [mpsGraph selectWithPredicateTensor:xEqualZeroPredicateTensor
                                                   truePredicateTensor:zeroTensor
                                                  falsePredicateTensor:xlogyTensor
                                                                  name:nil];
    outputTensor = [mpsGraph selectWithPredicateTensor:yIsNaNPredicateTensor
                                   truePredicateTensor:secondaryCastTensor
                                  falsePredicateTensor:outputTensor
                                                  name:nil];
    return outputTensor;
  };
  mps::binaryOpTensor(self, other, Scalar(1.0), output, "xlogy_out_mps", xlogy_op_block);
}

} // namespace at::native
