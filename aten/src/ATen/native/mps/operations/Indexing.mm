//  Copyright © 2022 Apple Inc.
#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/Utils.h>

#include <ATen/ceil_div.h>
#include <ATen/NativeFunctions.h>
#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/AccumulateType.h>
#include <torch/library.h>
#include <ATen/native/TensorAdvancedIndexing.h>
#include <ATen/native/IndexingUtils.h>
#include <c10/util/irange.h>
#include <c10/core/QScheme.h>
#include <c10/util/SmallVector.h>
#include <ATen/native/IndexKernel.h>

#ifdef __OBJC__
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

namespace at {
namespace native {

static
void getDataOffsets(const TensorIteratorBase& iter, simd_uint3* data_offsets, uint32_t num_outputs) {
  // Collect offsets for Input, Output and Index tensors
  const uint32_t nOffsets = 3;
  const uint32_t nDim = iter.ndim();

  std::vector<uint32_t> strides[nDim];
  const IntArrayRef& iterShape = iter.shape();

  for (uint32_t i=0; i < nDim; i++)
    for (uint32_t offset = 0; offset < nOffsets; offset++)
      strides[i].push_back(iter.strides(offset)[i]);

  for (int32_t i = 0; i < num_outputs; i++) {
    uint32_t idx = i;
    for (int dim = 0; dim < nDim; dim++) {
      uint32_t remainder = idx % iterShape[dim];
      idx /= iterShape[dim];

      for (int offset = 0; offset < nOffsets; offset++) {
        data_offsets[i][offset] += remainder * strides[dim][offset];
      }
    }
  }
}

static
bool dispatchIndexSelectKernel(TensorIteratorBase& iter, IntArrayRef index_size, IntArrayRef index_stride) {
  using namespace mps;

  @autoreleasepool {
    const Tensor& inputTensor = iter.tensor(1);
    Tensor outputTensor = iter.tensor(0);

    id<MTLBuffer> inputBuffer  = getMTLBufferStorage(inputTensor);
    id<MTLBuffer> outputBuffer = getMTLBufferStorage(outputTensor);
    MTLResourceOptions options = [inputBuffer resourceOptions];
    int64_t iter_numel = iter.numel();
    int64_t num_indices = index_size.size();

    std::string indexFunction;
    if (!selectIndexFunctionName(inputTensor.scalar_type(), indexFunction)) {
        TORCH_CHECK(false, "Failed to create indexing library, error: ", inputTensor.scalar_type());
    }

    NSError* error = nil;
    MPSStream* mpsStream = getCurrentMPSStream();
    id<MTLDevice> device = MPSDevice::getInstance()->device();

    id<MTLBuffer> offsets = [device newBufferWithLength: iter_numel * sizeof(simd_uint3)
                                                options: 0];
    // Get a list of offsets to index the Input, Output and the Index tensors
    simd_uint3* data_offsets = (simd_uint3*)[offsets contents];
    getDataOffsets(iter, data_offsets, iter_numel);

    id<MTLCommandBuffer> commandBuffer = mpsStream->commandBuffer();
    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

    MTLFunctionConstantValues* constantValues = [MTLFunctionConstantValues new];
    int64_t storage_offset = inputTensor.storage_offset();
    [constantValues setConstantValue: &storage_offset type:MTLDataTypeLong atIndex:0];
    [constantValues setConstantValue: &num_indices type:MTLDataTypeUInt atIndex:1];

    id<MTLFunction> indexKernelFunction = MPSDevice::getInstance()->metalFunction(indexFunction, constantValues);
    id<MTLArgumentEncoder> argumentEncoder = [indexKernelFunction newArgumentEncoderWithBufferIndex:0];
    NSUInteger argumentBufferLength = argumentEncoder.encodedLength;
    id<MTLBuffer> indexAB = [device newBufferWithLength:argumentBufferLength options:0];
    [argumentEncoder setArgumentBuffer:indexAB offset:0];

    for (uint32_t idx = 0; idx < num_indices; idx++) {
      const Tensor& indexTensor = iter.tensor(idx+2);
      [argumentEncoder setBuffer: getMTLBufferStorage(indexTensor)
                          offset: 0
                         atIndex: idx];
      TORCH_CHECK(indexTensor.scalar_type() == ScalarType::Long, "index(): Expected dtype int64 for index");
    }

    id<MTLBuffer> indexSize = [device newBufferWithBytes: index_size.data()
                                                  length: sizeof(index_size[0]) * index_size.size()
                                                 options: options];
    id<MTLBuffer> indexStride = [device newBufferWithBytes: index_stride.data()
                                                    length: sizeof(index_stride[0]) * index_stride.size()
                                                   options: options];

    // FIXME: PSO needs to be cached
    id<MTLComputePipelineState> indexSelectPSO = [device newComputePipelineStateWithFunction: indexKernelFunction
                                                                                       error: &error];
    TORCH_CHECK(indexSelectPSO, "Failed to created pipeline state object, error: ", [[error description] UTF8String]);

    for (uint32_t idx = 0; idx < num_indices; idx++) {
      const Tensor& indexTensor = iter.tensor(idx+2);
      [computeEncoder useResource:getMTLBufferStorage(indexTensor) usage:MTLResourceUsageRead];
    }

    [computeEncoder setComputePipelineState: indexSelectPSO];
    [computeEncoder setBuffer: indexAB offset:0 atIndex:0];
    [computeEncoder setBuffer: indexSize offset:0 atIndex:1];
    [computeEncoder setBuffer: indexStride offset:0 atIndex:2];
    [computeEncoder setBuffer: offsets offset:0 atIndex:3];
    [computeEncoder setBuffer: inputBuffer offset:0 atIndex:4];
    [computeEncoder setBuffer: outputBuffer offset:0 atIndex:5];

    MTLSize numOutputElems = MTLSizeMake(iter.numel(), 1, 1);
    NSUInteger threadGroupSize = indexSelectPSO.maxTotalThreadsPerThreadgroup;
    if (threadGroupSize > iter.numel()) {
        threadGroupSize = iter.numel();
    }

    MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);
    [computeEncoder dispatchThreads: numOutputElems
              threadsPerThreadgroup: threadgroupSize];

    [computeEncoder endEncoding];
    mpsStream->commit(true);
  }

  return true;
}

void index_kernel_mps(TensorIteratorBase& iter, IntArrayRef index_size, IntArrayRef index_stride) {
  using namespace mps;

  @autoreleasepool {
    int64_t num_indices = index_size.size();

    AT_ASSERT(num_indices == index_stride.size());
    AT_ASSERT(num_indices == iter.ntensors() - 2);
    const Tensor& inputTensor = iter.tensor(1);

    TORCH_CHECK(inputTensor.scalar_type() == ScalarType::Float ||
                inputTensor.scalar_type() == ScalarType::Half  ||
                inputTensor.scalar_type() == ScalarType::Long  ||
                inputTensor.scalar_type() == ScalarType::Int   ||
                inputTensor.scalar_type() == ScalarType::Short ||
                inputTensor.scalar_type() == ScalarType::Char  ||
                inputTensor.scalar_type() == ScalarType::Byte  ||
                inputTensor.scalar_type() == ScalarType::Bool, getMPSTypeString(inputTensor.scalar_type()) + std::string(" not supported for index.Tensor_out"));

    dispatchIndexSelectKernel(iter, index_size, index_stride);
  }
}

Tensor flip_mps(const Tensor& self, IntArrayRef dims) {
  using namespace mps;

  Tensor result = at::native::empty_mps(
                    self.sizes(),
                    self.scalar_type(),
                    c10::nullopt,
                    kMPS,
                    c10::nullopt,
                    c10::nullopt);

  auto total_dims = self.dim();
  // It wraps the dims and checks that there are no repeated dims
  auto flip_dims_b = at::dim_list_to_bitset(dims, total_dims);
  NSMutableArray<NSNumber*> * ns_dims = [NSMutableArray<NSNumber*> new];

  for (const auto i : c10::irange(total_dims)) {
    if(flip_dims_b[i] && self.size(i) > 1 && self.stride(i) != 0) {
      [ns_dims addObject:[NSNumber numberWithInt:i]];
    }
  }

  // Nothing to do, we return fast
  if (dims.size() == 0 || self.numel() <=1) {
    result.copy_(self);
    return result;
  }

  MPSStream* stream = getCurrentMPSStream();

  struct CachedGraph : public MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  MPSGraphCache* cache_ = MPSGraphCache::getInstance();

  @autoreleasepool {
    NSString* ns_dims_key = [[ns_dims valueForKey:@"description"] componentsJoinedByString:@","];
    // A key is used to identify the MPSGraph which was created once, and can be reused if the parameters, data types etc match the earlier created MPSGraph
    string key = "flip_mps:" + getTensorsStringKey({self}) + ":" + string([ns_dims_key UTF8String]);
    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));
    if(!cachedGraph) {
      MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {

        CachedGraph *newCachedGraph = nil;

        @autoreleasepool {
          MPSGraph* mpsGraph = make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
          MPSGraphTensor* outputTensor = [mpsGraph reverseTensor:inputTensor
                                                            axes:ns_dims
                                                            name:nil];
          newCachedGraph->inputTensor_ = inputTensor;
          newCachedGraph->outputTensor_ = outputTensor;
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }

    // Create placeholders which use the keys of the CachedGraph to create inputs and outputs of the operation
    Placeholder inputPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, result);


    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      inputPlaceholder.getMPSGraphTensor() : inputPlaceholder.getMPSGraphTensorData()
    };

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    // Run the graph
    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }

  return result;

}

Tensor index_select_mps(const Tensor & self,
                         int64_t dim,
                         const Tensor & index) {
  IntArrayRef input_shape = self.sizes();
  auto num_input_dims = input_shape.size();

  auto num_indices = index.numel();
  TORCH_CHECK_INDEX(index.dim() <= 1, "index_select(): Index is supposed to be a vector");

  dim = maybe_wrap_dim(dim, self.dim());
  std::vector<int64_t> shape_data(num_input_dims);

  // Calculate new shape
  for(auto i : c10::irange(num_input_dims)) {
    if (i == dim) {
      shape_data[i] = num_indices;
    } else {
      shape_data[i] = input_shape[i];
    }
  }

  IntArrayRef output_shape = IntArrayRef(shape_data.data(), num_input_dims);

  Tensor result = at::native::empty_mps(
                      output_shape,
                      self.scalar_type(),
                      c10::nullopt,
                      kMPS,
                      c10::nullopt,
                      c10::nullopt);

  index_select_out_mps(self, dim, index, result);
  return result;
}

Tensor& index_select_out_mps(const Tensor & self,
                             int64_t dim,
                             const Tensor & index,
                             Tensor & output) {

  using namespace mps;
  MPSStream* stream = getCurrentMPSStream();
  dim = maybe_wrap_dim(dim, self.dim());
  // Checks
  TORCH_CHECK_INDEX(index.dim() <= 1, "index_select(): Index is supposed to be a vector");
  TORCH_CHECK(index.scalar_type() == ScalarType::Long || index.scalar_type() == ScalarType::Int, "index_select(): Expected dtype int32 or int64 for index");
  TORCH_CHECK(self.scalar_type() == output.scalar_type(),
              "index_select(): self and output must have the same scalar type");
  TORCH_CHECK(dim == 0 || dim < self.dim(),
              "index_select(): Indexing dim ", dim, " is out of bounds of tensor");

  // Derive from MPSCachedGraph
  struct CachedGraph : public MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* inputTensor_ = nil;
    MPSGraphTensor* indexTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  MPSGraphCache* cache_ = MPSGraphCache::getInstance();

  @autoreleasepool {

    string key = "index_select_out_mps" + getTensorsStringKey({self, index}) + ":" + std::to_string(dim);
    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));

    if(!cachedGraph) {
      MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {
        CachedGraph *newCachedGraph = nil;

        @autoreleasepool {
          MPSGraph* mpsGraph = make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
          MPSGraphTensor* indexTensor = mpsGraphRankedPlaceHolder(mpsGraph, index);

          MPSGraphTensor* outputTensor = [mpsGraph gatherWithUpdatesTensor:inputTensor
                                                             indicesTensor:indexTensor
                                                                      axis:dim
                                                           batchDimensions:0
                                                                      name:nil];

          newCachedGraph->inputTensor_ = inputTensor;
          newCachedGraph->indexTensor_ = indexTensor;
          newCachedGraph->outputTensor_ = outputTensor;
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }

    Placeholder selfPlaceholder = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder indexPlaceholder = Placeholder(cachedGraph->indexTensor_, index);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      selfPlaceholder.getMPSGraphTensor() : selfPlaceholder.getMPSGraphTensorData(),
      indexPlaceholder.getMPSGraphTensor() : indexPlaceholder.getMPSGraphTensorData()
    };
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
  }

  return output;

}

Tensor & masked_fill__mps(Tensor& self, const Tensor & mask, const Scalar& value) {
  using namespace mps;
  TORCH_CHECK(self.device() == mask.device(), "expected self and mask to be on the same device, but got mask on ",
    mask.device(), " and self on ", self.device());
  TORCH_CHECK(mask.scalar_type() == kByte || mask.scalar_type() == kBool,
    "expected mask dtype to be Bool but got ", mask.scalar_type());
  auto maybe_outnames = namedinference::broadcast_to_outnames(self, mask, "masked_fill_");

  c10::MaybeOwned<Tensor> b_mask = expand_inplace(self, mask, "masked_fill_");

  struct CachedGraph : public MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor *inputTensor_ = nil;
    MPSGraphTensor *maskTensor_ = nil;
    MPSGraphTensor *outputTensor_ = nil;
  };

  MPSGraphCache* cache_ = MPSGraphCache::getInstance();

  MPSStream* stream = getCurrentMPSStream();
  @autoreleasepool {
    string key = "masked_fill" + getTensorsStringKey({self, mask}) + ":" + std::to_string(value.toDouble());
    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));
    if(!cachedGraph) {
      MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {

        CachedGraph *newCachedGraph = nil;

        @autoreleasepool {
          MPSGraph* mpsGraph = make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          MPSGraphTensor* inputTensor = mpsGraphRankedPlaceHolder(mpsGraph, self);
          MPSGraphTensor* maskTensor = mpsGraphRankedPlaceHolder(mpsGraph, mask);
          MPSDataType valueType = getMPSScalarType(value.type());

          // constantWithScalar doesn't like Bool constants getting created so
          // mapping them to int8
          if (valueType == MPSDataTypeBool) {
            valueType = MPSDataTypeInt8;
          }
          MPSGraphTensor* valueTensor =  [mpsGraph constantWithScalar:value.to<double>()
                                                            dataType:valueType];
          valueTensor = [mpsGraph castTensor:valueTensor
                                          toType:getMPSDataType(self.scalar_type())
                                           name : @"castTensorEq"];

          MPSGraphTensor* outputTensor = [mpsGraph selectWithPredicateTensor:maskTensor
                                                        truePredicateTensor:valueTensor
                                                        falsePredicateTensor:inputTensor
                                                             name:nil];

          newCachedGraph->inputTensor_ = inputTensor;
          newCachedGraph->maskTensor_ = maskTensor;
          newCachedGraph->outputTensor_ = outputTensor;
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }

    Placeholder selfPlaceholder   = Placeholder(cachedGraph->inputTensor_, self);
    Placeholder maskPlaceholder   = Placeholder(cachedGraph->maskTensor_, mask);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, self);

    // Create dictionary of inputs and outputs
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      selfPlaceholder.getMPSGraphTensor() : selfPlaceholder.getMPSGraphTensorData(),
      maskPlaceholder.getMPSGraphTensor() : maskPlaceholder.getMPSGraphTensorData()
    };

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    runMPSGraph(stream, cachedGraph->graph(), feeds, results);

  }
  namedinference::propagate_names_if_nonempty(self, maybe_outnames);
  return self;
}

Tensor embedding_dense_backward_mps(
    const Tensor & grad_, const Tensor & indices, int64_t num_weights,
    int64_t padding_idx, bool scale_grad_by_freq)
{
    // TODO: implement padding_idx & scale_grad_by_freq.
    namespace native_mps = at::native::mps;
    struct CachedGraph : public native_mps::MPSCachedGraph
    {
      CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
      MPSGraphTensor *incomingGradTensor_ = nil;
      MPSGraphTensor *indicesTensor_ = nil;
      MPSGraphTensor *outgoingGradTensor_ = nil;
    };

    native_mps::MPSGraphCache* cache_ = native_mps::MPSGraphCache::getInstance();

    IntArrayRef incoming_gradient_shape = grad_.sizes();
    int64_t num_incoming_gradient_dims = incoming_gradient_shape.size();

    IntArrayRef indices_shape = indices.sizes();
    int64_t num_indices_dims = indices_shape.size();

    int64_t D = incoming_gradient_shape[num_incoming_gradient_dims - 1];
    c10::SmallVector<int64_t, 2> outgoing_gradient_shape{num_weights, D};
    Tensor outgoing_gradient = at::native::empty_mps(
                                IntArrayRef(outgoing_gradient_shape.data(), outgoing_gradient_shape.size()),
                                grad_.scalar_type(),
                                c10::nullopt,
                                kMPS,
                                c10::nullopt,
                                c10::nullopt);

    if (outgoing_gradient.numel() == 0) {
      return outgoing_gradient;
    }

    auto stream = at::mps::getCurrentMPSStream();

    @autoreleasepool {
        string key = "edb_mps:" + native_mps::getMPSTypeString(grad_.scalar_type()) + ":indices" + std::to_string(num_indices_dims) + ":num_weights" + std::to_string(num_weights) + ":padding_idx" + std::to_string(padding_idx) + ":scaled" + std::to_string(scale_grad_by_freq);
      CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));
      // Initialize once if configuration not found in cache
      if(!cachedGraph) {
        native_mps::MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ native_mps::MPSCachedGraph * () {

          CachedGraph *newCachedGraph = nil;

          @autoreleasepool {
            MPSGraph* mpsGraph = native_mps::make_mps_graph();
            newCachedGraph = new CachedGraph(mpsGraph);

            MPSGraphTensor* incomingGradTensor = native_mps::mpsGraphUnrankedPlaceHolder(mpsGraph, native_mps::getMPSDataType(grad_.scalar_type()));

            MPSGraphTensor* indicesTensor = native_mps::mpsGraphUnrankedPlaceHolder(mpsGraph, native_mps::getMPSDataType(indices.scalar_type()));

            MPSGraphTensor *reshapedIndicesTensor = [mpsGraph  expandDimsOfTensor:indicesTensor
                             axes:@[@-1]
                             name:nil];

            MPSGraphTensor *outgoingGradTensor;
            outgoingGradTensor = [mpsGraph scatterNDWithUpdatesTensor:incomingGradTensor
                            indicesTensor:reshapedIndicesTensor
                                    shape:native_mps::getMPSShape(IntArrayRef(outgoing_gradient_shape.data(), outgoing_gradient_shape.size()))
                          batchDimensions:0
                                     mode:MPSGraphScatterModeAdd
                                     name:@"edb"];

            newCachedGraph->incomingGradTensor_ = incomingGradTensor;
            newCachedGraph->indicesTensor_ = indicesTensor;
            newCachedGraph->outgoingGradTensor_ = outgoingGradTensor;

          }
          return newCachedGraph;
        });
        cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
      }
      auto incomingGradPlaceholder = native_mps::Placeholder(cachedGraph->incomingGradTensor_, grad_);
      auto indicesPlaceholder = native_mps::Placeholder(cachedGraph->indicesTensor_, indices);
      auto outgoingGradPlaceholder = native_mps::Placeholder(cachedGraph->outgoingGradTensor_, outgoing_gradient);

      NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *feeds = @{
          incomingGradPlaceholder.getMPSGraphTensor() : incomingGradPlaceholder.getMPSGraphTensorData(),
          indicesPlaceholder.getMPSGraphTensor() : indicesPlaceholder.getMPSGraphTensorData()
      };

      NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *results = @{
          outgoingGradPlaceholder.getMPSGraphTensor() : outgoingGradPlaceholder.getMPSGraphTensorData()
      };
      native_mps::runMPSGraph(stream, cachedGraph->graph(), feeds, results);
    }
    return outgoing_gradient;
}

Tensor & masked_fill__mps(Tensor& self, const Tensor & mask, const Tensor & value) {
  TORCH_CHECK(value.dim() == 0, "masked_fill_ only supports a 0-dimensional value tensor, but got tensor "
      "with ", value.dim(), " dimension(s).");
  return masked_fill__mps(self, mask, value.item());
}

REGISTER_DISPATCH(index_stub, &index_kernel_mps);

} // native
} // at
