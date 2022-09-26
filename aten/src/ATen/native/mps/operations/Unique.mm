//  Copyright © 2022 Apple Inc.

#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/mps/MPSAllocator.h>

namespace at {
namespace native {
namespace mps {

struct UniqueCachedGraph : public MPSCachedGraph
{
  UniqueCachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
  MPSGraphTensor* inputTensor_ = nil;
  MPSGraphTensor* outputTensor_ = nil;
  MPSGraphTensor* inverseIndicesTensor_ = nil;
  MPSGraphTensor* countsTensor_ = nil;
  MPSGraphTensor* lengthTensor_ = nil;
};

struct UniqueSliceCachedGraph : public MPSCachedGraph
{
  UniqueSliceCachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
  MPSGraphTensor* outputTensor_ = nil;
};

const char * dataTypeToString(MPSDataType dataType)
{
    const char * result = nil;
    switch (dataType)
    {
        case MPSDataTypeInvalid: result = "MPSDataTypeInvalid"; break;
        case MPSDataTypeFloatBit: result = "MPSDataTypeFloatBit"; break;
        case MPSDataTypeFloat32: result = "MPSDataTypeFloat32"; break;
        case MPSDataTypeFloat16: result = "MPSDataTypeFloat16"; break;
        case MPSDataTypeSignedBit: result = "MPSDataTypeSignedBit"; break;
        case MPSDataTypeInt8: result = "MPSDataTypeInt8"; break;
        case MPSDataTypeInt16: result = "MPSDataTypeInt16"; break;
        case MPSDataTypeInt32: result = "MPSDataTypeInt32"; break;
        case MPSDataTypeInt64: result = "MPSDataTypeInt64"; break;
        case MPSDataTypeUInt8: result = "MPSDataTypeUInt8"; break;
        case MPSDataTypeUInt16: result = "MPSDataTypeUInt16"; break;
        case MPSDataTypeUInt32: result = "MPSDataTypeUInt32"; break;
        case MPSDataTypeUInt64: result = "MPSDataTypeUInt64"; break;
        case MPSDataTypeNormalizedBit: result = "MPSDataTypeNormalizedBit"; break;
        case MPSDataTypeUnorm1: result = "MPSDataTypeUnorm1"; break;
        case MPSDataTypeUnorm8: result = "MPSDataTypeUnorm8"; break;
        default: result = "<Unknown datatype>"; break;
    }
    return result;
}

static std::string getUniqueKey(const ScalarType& dtype, const IntArrayRef& base_shape,
                                const bool return_inverse, const bool return_counts,
                                const bool consecutive, c10::optional<int64_t> dimOpt)
{
  return "_unique2_mps:" + getMPSTypeString(dtype) + "[" + getArrayRefString(base_shape) +
         "]:[" + (dimOpt.has_value() ? to_string(dimOpt.value()) : "None") + "]:[" + to_string(return_inverse) +
         "]:[" + to_string(return_counts) + "]:[" + to_string(consecutive) + "]";
}

NSArray<MPSGraphTensor*> *buildUniqueGraph(UniqueCachedGraph *uniqueGraph, const bool return_inverse, const bool return_counts, const bool consecutive, c10::optional<int64_t> dimOpt) {
  int64_t dim = dimOpt.has_value() ? dimOpt.value() : 0;
  
  MPSGraph *graph = uniqueGraph->graph();
  MPSGraphTensor *inputTensor = uniqueGraph->inputTensor_;
  MPSShape *shape = [inputTensor shape];
  NSUInteger length = [shape[dim] integerValue];
  MPSDataType dataType = [inputTensor dataType];
  
  MPSGraphTensor *resultTensor = nil;
  MPSGraphTensor *inverseIndicesTensor = nil;
  MPSGraphTensor *countTensor = nil;
  MPSGraphTensor *lengthTensor = nil;
  if (length <= 1) {
    return @[resultTensor, inverseIndicesTensor, countTensor, lengthTensor];
  }
  
  // Sort only supports following types, cast if necessary
  if (dataType != MPSDataTypeInt32 &&
      dataType != MPSDataTypeFloat32 &&
      dataType != MPSDataTypeFloat16) {
    dataType = (dataType & MPSDataTypeFloatBit) ? MPSDataTypeFloat32 : MPSDataTypeInt32;
    inputTensor = [graph castTensor:inputTensor
                             toType:dataType
                               name:@"castInputTensor"];
  }

  if (!dimOpt.has_value())
    inputTensor = [graph reshapeTensor:inputTensor
                             withShape:@[@-1]
                                  name:nil];

  MPSGraphTensor *sortedInput;
  MPSGraphTensor *argSortedInput;
  if (consecutive) {
    sortedInput = inputTensor;
    argSortedInput = [graph coordinateAlongAxis:dim];
  } else {
    sortedInput = [graph sortWithTensor:inputTensor
                                   axis:dim
                                   name:nil];
    argSortedInput = [graph argSortWithTensor:inputTensor
                                         axis:dim
                                         name:nil];
  }
  
  MPSGraphTensor *frontNMinusOne = [graph sliceTensor:sortedInput
                                            dimension:dim
                                                start:0
                                               length:length-1
                                                 name:nil];
  MPSGraphTensor *backNMinusOne = [graph sliceTensor:sortedInput
                                           dimension:dim
                                               start:1
                                              length:length-1
                                                name:nil];
  MPSGraphTensor *notEqualToPreviousElement = [graph notEqualWithPrimaryTensor:backNMinusOne
                                                               secondaryTensor:frontNMinusOne
                                                                          name:nil];
  MPSGraphTensor *castedMask = [graph castTensor:notEqualToPreviousElement
                                          toType:MPSDataTypeInt32
                                            name:@"castMaskTensor"];
  MPSGraphTensor *scannedIndices = [graph cumulativeSumWithTensor:castedMask
                                                             axis:dim
                                                             name:nil];
  lengthTensor = [graph sliceTensor:scannedIndices
                          dimension:dim
                              start:length-2
                             length:1
                               name:nil];
  if ([shape count] > 1) {
      lengthTensor = [graph reductionMaximumWithTensor:lengthTensor
                                                  axes:nil
                                                  name:nil];
  }
  MPSGraphTensor *minusOneTensor = [graph constantWithScalar:-1.0f
                                                    dataType:MPSDataTypeInt32];
  MPSGraphTensor *maskedIndices = [graph selectWithPredicateTensor:notEqualToPreviousElement
                                               truePredicateTensor:scannedIndices
                                              falsePredicateTensor:minusOneTensor
                                                              name:nil];
  NSMutableArray *headShape = [[NSMutableArray alloc] initWithArray:shape];
  headShape[dim] = @1;
  MPSGraphTensor *zeroTensor = [graph constantWithScalar:0.0f
                                                   shape:headShape
                                                dataType:MPSDataTypeInt32];
  [headShape release];
  MPSGraphTensor *maskedIndicesWithHead = [graph concatTensors:@[zeroTensor, maskedIndices]
                                                     dimension:dim
                                                          name:nil];
  MPSGraphTensor *scannedIndicesWithHead = [graph concatTensors:@[zeroTensor, scannedIndices]
                                                      dimension:dim
                                                           name:nil];
  
  resultTensor = [graph scatterAlongAxisWithUpdatesTensor:sortedInput
                                            indicesTensor:maskedIndicesWithHead
                                                    shape:shape
                                                     axis:dim
                                                     mode:MPSGraphScatterModeSet
                                                     name:nil];
  
  inverseIndicesTensor = [graph scatterAlongAxisWithUpdatesTensor:scannedIndicesWithHead
                                                    indicesTensor:argSortedInput
                                                            shape:shape
                                                             axis:dim
                                                             name:nil];

  MPSGraphTensor *unitTensor = [graph constantWithScalar:1.0f
                                                   shape:shape
                                                dataType:MPSDataTypeInt64];

  countTensor = [graph scatterAlongAxisWithUpdatesTensor:unitTensor
                                           indicesTensor:scannedIndicesWithHead
                                                   shape:shape
                                                    axis:dim
                                                    name:nil];
  
  // Cast back if necessary
  if ([uniqueGraph->inputTensor_ dataType] != dataType)
    resultTensor = [graph castTensor:resultTensor
                              toType:[uniqueGraph->inputTensor_ dataType]
                                name:@"castResultTensor"];
  
  return @[resultTensor, inverseIndicesTensor, countTensor, lengthTensor];
}

static UniqueCachedGraph* getUniqueGraph(const Tensor& self, const bool return_inverse, const bool return_counts, const bool consecutive, c10::optional<int64_t> dim) {
  MPSGraphCache* cache_ = MPSGraphCache::getInstance();

  @autoreleasepool {
    string key = getUniqueKey(self.scalar_type(), self.sizes(), return_inverse, return_counts, consecutive, dim);
    UniqueCachedGraph* cachedGraph = static_cast<UniqueCachedGraph *>(cache_->LookUp(key));
    if(!cachedGraph) {
      MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {

        UniqueCachedGraph *newCachedGraph = nil;

         @autoreleasepool {
           // Initialize graph
           MPSGraph* mpsGraph = make_mps_graph();
           newCachedGraph = new UniqueCachedGraph(mpsGraph);
           
           // Workaround for MPSShaderLibrary bug
           // TODO: Remove once https://github.com/pytorch/pytorch/issues/82305 is resolved
           auto inputType = getMPSScalarType(self.scalar_type());
           if (inputType ==  MPSDataTypeUInt8) {
               inputType =  MPSDataTypeInt8;
           }
           newCachedGraph->inputTensor_ = mpsGraphRankedPlaceHolder(mpsGraph, inputType, getMPSShape(self.sizes()));

           NSArray<MPSGraphTensor *> *outputTensors = buildUniqueGraph(newCachedGraph, return_inverse, return_counts, consecutive, dim);

           newCachedGraph->outputTensor_ = outputTensors[0];
           newCachedGraph->inverseIndicesTensor_ = outputTensors[1];
           newCachedGraph->countsTensor_ = outputTensors[2];
           newCachedGraph->lengthTensor_ = outputTensors[3];
         }
         return newCachedGraph;
       });
       cachedGraph = static_cast<UniqueCachedGraph *>(tmpCachedGraph);
     }
    return cachedGraph;
  }
}

void runUniqueGraph(UniqueCachedGraph *uniqueGraph, const Tensor& input, Tensor& output,
                    Tensor& inverse_indices, Tensor& counts, Tensor& length){
  Placeholder inputPlaceholder = Placeholder(uniqueGraph->inputTensor_, input);
  Placeholder outputPlaceholder = Placeholder(uniqueGraph->outputTensor_, output);
  Placeholder inverseIndicesPlaceholder = Placeholder(uniqueGraph->inverseIndicesTensor_, inverse_indices);
  Placeholder countsPlaceholder = Placeholder(uniqueGraph->countsTensor_, counts);
  Placeholder lengthPlaceholder = Placeholder(uniqueGraph->lengthTensor_, length);

  NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
    inputPlaceholder.getMPSGraphTensor() : inputPlaceholder.getMPSGraphTensorData(),
  };

  NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
    outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData(),
    inverseIndicesPlaceholder.getMPSGraphTensor() : inverseIndicesPlaceholder.getMPSGraphTensorData(),
    countsPlaceholder.getMPSGraphTensor() : countsPlaceholder.getMPSGraphTensorData(),
    lengthPlaceholder.getMPSGraphTensor() : lengthPlaceholder.getMPSGraphTensorData(),
  };

  // Run the graph
  MPSStream* stream = getCurrentMPSStream();
  runMPSGraph(stream, uniqueGraph->graph(), feeds, results);
}

} // namespace mps

std::tuple<Tensor, Tensor, Tensor>
_unique_impl_mps(const Tensor& self, const bool return_inverse, const bool return_counts, const bool consecutive, c10::optional<int64_t> dimOpt) {
  printf("Running mps unique.\n");
  
  const Tensor& input = self.contiguous();

  Tensor output = at::native::empty_mps(input.sizes(), input.scalar_type(), c10::nullopt, kMPS);
  Tensor inverse_indices = at::native::empty_mps(input.sizes(), ScalarType::Long, c10::nullopt, kMPS);
  Tensor counts = at::native::empty_mps(input.sizes(), ScalarType::Long, c10::nullopt, kMPS);
  Tensor length = at::native::empty_mps({1}, ScalarType::Int, c10::nullopt, kMPS);

  if (input.numel() == 0)
    return std::make_tuple(output, inverse_indices, counts);

  mps::UniqueCachedGraph *uniqueGraph = mps::getUniqueGraph(input, return_inverse, return_counts, consecutive, dimOpt);
//  NSLog(@"%@", [uniqueGraph->graph() debugDescription]);

  mps::runUniqueGraph(uniqueGraph, input, output, inverse_indices, counts, length);

  int64_t lengthScalar = length.item<int64_t>() + 1; // length actually holds max index, add 1

//  printf("length is: %lld\n", lengthScalar);
  int64_t dim = dimOpt.has_value() ? dimOpt.value() : 0;
//  printf("dim is: %lld\n", dim);

  output = at::slice(output, dim, 0, lengthScalar);
  counts = at::slice(counts, dim, 0, lengthScalar);
  
  printf("Finished running mps unique.\n");

  return std::make_tuple(output, inverse_indices, counts);
}

std::tuple<Tensor, Tensor, Tensor>
unique_dim_mps(const Tensor& self, int64_t dim, const bool sorted, const bool return_inverse, const bool return_counts) {
  printf("unique_dim\n");
  return _unique_impl_mps(self, return_inverse, return_counts, false, c10::make_optional((int64_t)dim));
}

std::tuple<Tensor, Tensor, Tensor>
unique_consecutive_mps(const Tensor& self, const bool return_inverse, const bool return_counts, c10::optional<int64_t> dim) {
  printf("unique_consecutive\n");
  return _unique_impl_mps(self, return_inverse, return_counts, true, dim);
}

std::tuple<Tensor, Tensor, Tensor>
unique_dim_consecutive_mps(const Tensor& self, int64_t dim, const bool return_inverse, const bool return_counts) {
  printf("unique_dim_consecutive\n");
  return _unique_impl_mps(self, return_inverse, return_counts, true, c10::make_optional((int64_t)dim));
}

std::tuple<Tensor, Tensor, Tensor>
_unique2_mps(const Tensor& self, const bool sorted, const bool return_inverse, const bool return_counts) {
  printf("unique2\n");
  return _unique_impl_mps(self, return_inverse, return_counts, false, c10::nullopt);
}

} // namespace native
} // namespace at
