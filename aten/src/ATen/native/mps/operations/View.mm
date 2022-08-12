//  Copyright © 2022 Apple Inc.

#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/Resize.h>

#define ENABLE_LOG 0
#define ENABLE_VIEW_KERNELS 1
#define SHOW_NOT_SUPPORTED_VIEWS 0

#define COMMIT_AND_CONTINUE   1

namespace at {

// these are from MPSAllocator
namespace mps {
  // to check the requested non-aligned size of an MTL buffer
  ssize_t get_requested_buffer_size(void* ptr);
  // to retrieve the shape of a base tensor from a view tensor
  IntArrayRef get_buffer_shape(void* ptr);
  // to set the shape of a base tensor from a view tensor
  void set_buffer_shape(void* ptr, const IntArrayRef& shape);
}

namespace native {
namespace mps {

enum class ScatterGatherOpViewType {
  MTL_SCATTER,
  MTL_GATHER,
  MTL_GATHER_SCATTER
};

struct ViewCachedGraph : public MPSCachedGraph
{
  ViewCachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
  MPSGraphTensor* inputTensor = nil;
  MPSGraphTensor* outputTensor = nil;
  MPSGraphTensor* updatesTensor = nil;
  MPSGraphTensor* storageOffsetTensor = nil;
  std::vector<MPSGraphTensor*> strideTensors;
};

static std::string getStridedKey(const ScalarType& dtype, const IntArrayRef& base_shape,
                          const IntArrayRef& new_shape, bool is_scatter)
{
  return (is_scatter ? "scatter:" : "gather:") + getMPSTypeString(dtype) + "[" +
         getArrayRefString(base_shape) + "]:[" + getArrayRefString(new_shape) + "]";
}

// initializes the MTLBuffers for tesnsor data and runs the MPSGraph for the view op
static Tensor& runViewGraph(ViewCachedGraph* cachedGraph, const at::Tensor& src, Tensor& output,
                            bool needsScatter, bool requires_sync = false)
{
  const id<MTLBuffer> sourceBuffer = getMTLBufferStorage(src);
  const id<MTLBuffer> outputBuffer = getMTLBufferStorage(output);

  const IntArrayRef& strides   = needsScatter ? output.strides() : src.strides();
  const IntArrayRef& sizes     = needsScatter ? output.sizes() : src.sizes();
  const int64_t storage_offset = needsScatter ? output.storage_offset() : src.storage_offset();
  const MPSDataType inputType  = [cachedGraph->inputTensor dataType];

  MPSShape *inputShape = [cachedGraph->inputTensor shape];
  MPSShape *outputShape = needsScatter ? inputShape : getMPSShape(src);

  MPSStream* stream = getCurrentMPSStream();
  @autoreleasepool {
    NSMutableDictionary *feeds = [[NSMutableDictionary new] autorelease];
    // in case of scatter, we use ouput tensor as input buffer and write the results back to the source buffer
    feeds[cachedGraph->inputTensor] = [[[MPSGraphTensorData alloc] initWithMTLBuffer: needsScatter ? outputBuffer : sourceBuffer
                                                                               shape: inputShape
                                                                            dataType: inputType] autorelease];
    if (needsScatter) {
      feeds[cachedGraph->updatesTensor] = [[[MPSGraphTensorData alloc] initWithMTLBuffer: sourceBuffer
                                                                                   shape: getMPSShape(src.numel())
                                                                                dataType: inputType] autorelease];
    }
    feeds[cachedGraph->storageOffsetTensor] = getMPSGraphTensorFromScalar(stream, Scalar(storage_offset), MPSDataTypeInt32);
    for (int i = 0; i < sizes.size(); i++) {
      feeds[cachedGraph->strideTensors[i]] = getMPSGraphTensorFromScalar(stream, Scalar(strides[i]), MPSDataTypeInt32);
    }
    MPSGraphTensorData* outputTensorData = [[[MPSGraphTensorData alloc] initWithMTLBuffer: outputBuffer
                                                                                    shape: outputShape
                                                                                 dataType: getMPSDataType(output.scalar_type())] autorelease];
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      cachedGraph->outputTensor : outputTensorData
    };
    stream->executeMPSGraph(cachedGraph->graph(), feeds, results,
                            requires_sync ? SyncType::COMMIT : SyncType::NONE);
  }
  return output;
}

static MPSGraphTensor* chainViewOperation(ViewCachedGraph* cachedGraph, const IntArrayRef& size,
                                          const IntArrayRef& stride, int64_t offset,
                                          const IntArrayRef& base_shape, bool needsScatter)
{
  MPSGraph* mpsGraph = cachedGraph->graph();
  MPSGraphTensor *outputTensor = nil;
  const size_t shape_size = size.size();

  @autoreleasepool {
    std::vector<int32_t> sizeArray(shape_size);
    const int64_t int_max = std::numeric_limits<int32_t>::max();
    for (int i = 0; i < shape_size; i++) {
      TORCH_CHECK(size[i] <= int_max);
      sizeArray[i] = static_cast<int32_t>(size[i]);
    }
    NSData* shapeData = [NSData dataWithBytes: sizeArray.data()
                                       length: shape_size * sizeof(int32_t)];
    MPSGraphTensor* shapeTensor = [mpsGraph constantWithData: shapeData
                                                       shape: @[[NSNumber numberWithUnsignedInteger: shape_size]]
                                                    dataType: MPSDataTypeInt32];
    MPSGraphTensor* indicesTensor = nil;
    // create stride Tensors for each rank of the input tensor
    for (int i = 0; i < shape_size; i++) {
      MPSGraphTensor* rangeTensor = [mpsGraph coordinateAlongAxis: (-i - 1)
                                                  withShapeTensor: shapeTensor
                                                             name: nil];
      MPSGraphTensor* strideTensor = cachedGraph->strideTensors[shape_size - i - 1];
      MPSGraphTensor* indexTensor = [mpsGraph multiplicationWithPrimaryTensor: rangeTensor
                                                              secondaryTensor: strideTensor
                                                                         name: nil];
      if (!indicesTensor) {
        indicesTensor = indexTensor;
      } else {
        indicesTensor = [mpsGraph additionWithPrimaryTensor: indexTensor
                                            secondaryTensor: indicesTensor
                                                       name: nil];
      }
    }

    indicesTensor = [mpsGraph additionWithPrimaryTensor: indicesTensor
                                        secondaryTensor: cachedGraph->storageOffsetTensor
                                                   name: nil];
    MPSGraphTensor *reshapedInputTensor = [mpsGraph reshapeTensor: cachedGraph->inputTensor
                                                        withShape: @[@-1]
                                                             name: nil];
    MPSGraphTensor *reshapedIndicesTensor = [mpsGraph reshapeTensor: indicesTensor
                                                          withShape: @[@-1]
                                                               name: nil];
    if (needsScatter) {
      MPSGraphTensor* scatteredTensor = [mpsGraph scatterAlongAxis: 0
                                                    withDataTensor: reshapedInputTensor
                                                     updatesTensor: cachedGraph->updatesTensor
                                                     indicesTensor: reshapedIndicesTensor
                                                              mode: MPSGraphScatterModeSet
                                                              name: nil];
      outputTensor = [mpsGraph reshapeTensor: scatteredTensor
                                   withShape: getMPSShape(base_shape)
                                        name: nil];
    } else {
      // Call gather to coalesce the needed values. Result will be of same shape as flattened indices tensor
      MPSGraphTensor *gatheredTensor = [mpsGraph gatherWithUpdatesTensor: reshapedInputTensor
                                                           indicesTensor: reshapedIndicesTensor
                                                                    axis: 0
                                                         batchDimensions: 0
                                                                    name: nil];
      // Reshape the data to desired size
      outputTensor =  [mpsGraph reshapeTensor: gatheredTensor
                              withShapeTensor: shapeTensor
                                         name: nil];
    }
  }
  return outputTensor;
}

// There are few cases we need to consider:
// Here nodes are the Tensors and the edges are the operations performed on the
// Tensor. As a result of the operation performed we can have result as View
// Tensor (View T) or a Non view tensor (NonView T). The difference is if its
// mapped by the same underlying storage ptr or a new MTLBuffer was allocated.
//                T = Tensor
//                 ----------
//                 | Orig T |
//                 ----------
//                /     |     \
//             View T  View T  NonView T
//             /      /    \      |
//            View T /      \     |
//            |     /        \    |
//            |    /          \   |
//            |   /            \  |
//            NonView T         NonView T
static ViewCachedGraph* createViewGraph(const Tensor& self, IntArrayRef size, IntArrayRef stride, int64_t storage_offset, bool needsScatter)
{
  IntArrayRef base_shape = get_buffer_shape(self.storage().data());
  if (base_shape.size() == 0) {
    // IntArrayRef wouldn't own the data, so we use a static storage
    static const int64_t shape_1d = 1;
    // self.sizes().size() could be zero
    base_shape = self.sizes().size() ? self.sizes() : IntArrayRef(&shape_1d, 1);
    // base_shape will be retained in MPSAllocator until buffer gets recycled
    if (self.storage().data())
      set_buffer_shape(self.storage().data(), base_shape);
  }

  return nil;
}

static
std::string getGatherScatterFunctionName(
  ScalarType scalarType,
  const IntArrayRef& sizes,
  const IntArrayRef& strides,
  ScatterGatherOpViewType gatherScatterViewType) {
  assert(sizes.size() == strides.size());
  std::string kernelName = (gatherScatterViewType == ScatterGatherOpViewType::MTL_GATHER)  ? "gather" :
                           (gatherScatterViewType == ScatterGatherOpViewType::MTL_SCATTER) ? "scatter" : "gather_scatter";

  kernelName += "_kernel_";
  std::string metalDType = getMetalScalarType(scalarType);
  TORCH_CHECK(!metalDType.empty(), "Unsupported data type");
  return kernelName + metalDType + std::to_string(sizes.size());
}

Tensor gatherViewTensor(const at::Tensor& src, at::Tensor& dst) {
  ViewCachedGraph* cachedGraph = nullptr;
  const IntArrayRef& base_shape = get_buffer_shape(src.storage().data());
  if (base_shape.size() == 0) return Tensor();

  id<MTLBuffer> outputBuffer;
  Tensor output;
  int64_t outputStorageOffset = 0;
  int64_t numThreads;

  if (!dst.has_storage()) {
    output = at::native::empty_mps(src.sizes(), src.scalar_type(), c10::nullopt, kMPS);
    outputBuffer = getMTLBufferStorage(output);
    numThreads = output.numel();
  } else {
    outputBuffer = getMTLBufferStorage(dst);
    outputStorageOffset = dst.storage_offset() * dst.element_size();
    numThreads = dst.numel();
    output = dst;
  }

  NSError *error = nil;
  MPSStream* mpsStream = getCurrentMPSStream();

  dispatch_sync(mpsStream->queue(), ^(){
    id<MTLComputeCommandEncoder> computeEncoder = [mpsStream->commandBuffer() computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    std::string functionName = getGatherScatterFunctionName(output.scalar_type(), output.sizes(), output.strides(), ScatterGatherOpViewType::MTL_GATHER);
    id<MTLComputePipelineState> gatherPSO = MPSDevice::getInstance()->metalPSO(functionName);

    uint32_t kernel_size = src.sizes().size();
    uint32_t src_sizes[kernel_size];
    uint32_t src_strides[kernel_size];

    for (int i = 0; i < kernel_size; i++) {
      src_sizes[i] = (uint32_t)(src.sizes()[i]);
      src_strides[i] = (uint32_t)(src.strides()[i]);
    }

    [computeEncoder setComputePipelineState: gatherPSO];
    [computeEncoder setBuffer:getMTLBufferStorage(src) offset:src.storage_offset() * src.element_size() atIndex:0];
    [computeEncoder setBuffer:outputBuffer offset:outputStorageOffset atIndex:1];
    [computeEncoder setBytes:&src_sizes[0] length:sizeof(uint32_t) * kernel_size atIndex:2];
    [computeEncoder setBytes:&src_strides[0] length:sizeof(uint32_t) * kernel_size atIndex:3];
    [computeEncoder setBytes:&numThreads length:sizeof(uint32_t) atIndex:4];

    MTLSize gridSize = MTLSizeMake(numThreads > 32 ? 32 : numThreads,
                                   numThreads / 32 > 32 ? 32 : (numThreads > 32 ? numThreads / 32 + 1 : 1),
                                   numThreads > 1024 ? numThreads / 1024 + 1 : 1);

    NSUInteger w = gatherPSO.threadExecutionWidth;
    NSUInteger h = gatherPSO.maxTotalThreadsPerThreadgroup / w;

    MTLSize threadsPerThreadgroup = MTLSizeMake(w, h, 1);


    [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadsPerThreadgroup];
    [computeEncoder endEncoding];
#if COMMIT_AND_CONTINUE
    mpsStream->commitAndContinue();
#else
    mpsStream->synchronize(SyncType::COMMIT);
#endif
  });

  return (dst.has_storage()) ? dst : output;
}

Tensor& scatterViewTensor(const at::Tensor& src, at::Tensor& output) {
  id<MTLBuffer> outputBuffer = getMTLBufferStorage(output);
  int64_t numThreads = src.numel();
  int64_t outputStorageOffset = output.storage_offset() * output.element_size();
  NSError *error = nil;
  MPSStream* mpsStream = getCurrentMPSStream();

  dispatch_sync(mpsStream->queue(), ^(){
    @autoreleasepool {
      // std::cout << "scatter:(" << src.numel() << ", " << output.numel() << "), (" << output.sizes() << "), (" << output.strides() << ")" << std::endl;

      id<MTLCommandBuffer> commandBuffer = mpsStream->commandBuffer();
      id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
      std::string functionName = getGatherScatterFunctionName(output.scalar_type(), output.sizes(), output.strides(), ScatterGatherOpViewType::MTL_SCATTER);
      id<MTLComputePipelineState> scatterPSO = MPSDevice::getInstance()->metalPSO(functionName);

      uint32_t kernel_size = output.sizes().size();
      uint32_t output_sizes[kernel_size];
      uint32_t output_strides[kernel_size];

      for (int i = 0; i < kernel_size; i++) {
        output_sizes[i] = (uint32_t)(output.sizes()[i]);
        output_strides[i] = (uint32_t)(output.strides()[i]);
      }

      [computeEncoder setComputePipelineState: scatterPSO];
      [computeEncoder setBuffer:getMTLBufferStorage(src) offset:src.storage_offset() * src.element_size() atIndex:0];
      [computeEncoder setBuffer:outputBuffer offset:outputStorageOffset atIndex:1];
      [computeEncoder setBytes:&output_sizes[0] length:sizeof(uint32_t) * kernel_size atIndex:2];
      [computeEncoder setBytes:&output_strides[0] length:sizeof(uint32_t) * kernel_size atIndex:3];
      [computeEncoder setBytes:&numThreads length:sizeof(uint32_t) atIndex:4];

      MTLSize gridSize = MTLSizeMake(numThreads > 32 ? 32 : numThreads,
                                     numThreads / 32 > 32 ? 32 : (numThreads > 32 ? numThreads / 32 + 1 : 1),
                                     numThreads > 1024 ? numThreads / 1024 + 1 : 1);

      NSUInteger w = scatterPSO.threadExecutionWidth;
      NSUInteger h = scatterPSO.maxTotalThreadsPerThreadgroup / w;
      MTLSize threadsPerThreadgroup = MTLSizeMake(w, h, 1);

      [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadsPerThreadgroup];
      [computeEncoder endEncoding];
#if COMMIT_AND_CONTINUE
      mpsStream->commitAndContinue();
#else
      mpsStream->synchronize(SyncType::COMMIT);
#endif
    }
  });

  return output;
}

Tensor& gatherScatterViewTensor(const at::Tensor& src, at::Tensor& output) {
  id<MTLBuffer> outputBuffer = getMTLBufferStorage(output);
  int64_t numThreads = output.numel();
  int64_t outputStorageOffset = output.storage_offset() * output.element_size();
  NSError *error = nil;
  MPSStream* mpsStream = getCurrentMPSStream();

  dispatch_sync(mpsStream->queue(), ^(){
    @autoreleasepool {
      id<MTLCommandBuffer> commandBuffer = mpsStream->commandBuffer();
      id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
      std::string functionName = getGatherScatterFunctionName(output.scalar_type(), output.sizes(), output.strides(), ScatterGatherOpViewType::MTL_GATHER_SCATTER);
      id<MTLComputePipelineState> gatherScatterPSO = MPSDevice::getInstance()->metalPSO(functionName);

      uint32_t kernel_size = output.sizes().size();
      uint32_t output_sizes[kernel_size];
      uint32_t output_strides[kernel_size];
      uint32_t src_sizes[kernel_size];
      uint32_t src_strides[kernel_size];

      for (int i = 0; i < kernel_size; i++) {
        output_sizes[i] = (uint32_t)(output.sizes()[i]);
        output_strides[i] = (uint32_t)(output.strides()[i]);
        src_sizes[i] = (uint32_t)(src.sizes()[i]);
        src_strides[i] = (uint32_t)(src.strides()[i]);
      }

      [computeEncoder setComputePipelineState: gatherScatterPSO];
      [computeEncoder setBuffer:getMTLBufferStorage(src) offset:src.storage_offset() * src.element_size() atIndex:0];
      [computeEncoder setBuffer:outputBuffer offset:outputStorageOffset atIndex:1];
      [computeEncoder setBytes:&output_sizes[0] length:sizeof(uint32_t) * kernel_size atIndex:2];
      [computeEncoder setBytes:&output_strides[0] length:sizeof(uint32_t) * kernel_size atIndex:3];
      [computeEncoder setBytes:&src_sizes[0] length:sizeof(uint32_t) * kernel_size atIndex:4];
      [computeEncoder setBytes:&src_strides[0] length:sizeof(uint32_t) * kernel_size atIndex:5];
      [computeEncoder setBytes:&numThreads length:sizeof(uint32_t) atIndex:6];

      MTLSize gridSize = MTLSizeMake(numThreads > 32 ? 32 : numThreads,
                                     numThreads / 32 > 32 ? 32 : (numThreads > 32 ? numThreads / 32 + 1 : 1),
                                     numThreads > 1024 ? numThreads / 1024 + 1 : 1);

      NSUInteger w = gatherScatterPSO.threadExecutionWidth;
      NSUInteger h = gatherScatterPSO.maxTotalThreadsPerThreadgroup / w;
      MTLSize threadsPerThreadgroup = MTLSizeMake(w, h, 1);

      [computeEncoder dispatchThreads:gridSize threadsPerThreadgroup:threadsPerThreadgroup];
      [computeEncoder endEncoding];
#if COMMIT_AND_CONTINUE
      mpsStream->commitAndContinue();
#else
      mpsStream->synchronize(SyncType::COMMIT);
#endif
    }
  });

  return output;
}



} // namespace mps

// implementation of as_strided() op
Tensor as_strided_tensorimpl_mps(const Tensor& self, IntArrayRef size, IntArrayRef stride, optional<int64_t> storage_offset_)
{
  auto storage_offset = storage_offset_.value_or(self.storage_offset());
  auto result = detail::make_tensor<TensorImpl>(c10::TensorImpl::VIEW, Storage(self.storage()), self.key_set(), self.dtype());
  setStrided(result, size, stride, storage_offset);

  // 0 sizes won't result in any change in the shape of the Tensor so we can skip it.
  if (size.size() > 0) {
    mps::createViewGraph(self, size, stride, storage_offset, /*needsScatter*/ false);
  }
  return result;
}

} // namespace native
} // namespace at
