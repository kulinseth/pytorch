//  Copyright © 2022 Apple Inc.

#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/TensorFactories.h>

namespace at {
namespace native {

Tensor& randperm_out_mps(int64_t n, c10::optional<Generator> generator, Tensor& result) {
  if (!MPSDevice::getInstance()->macOS_13_0()) {
    TORCH_WARN_ONCE("MPS: randperm op is supported natively starting from macOS 13.0. ",
                    "Falling back on CPU. This may have performace implications.");

    result = result.to("cpu");
    result = at::randperm_out(result, n).to("mps");
    return result;
  }

  using namespace mps;
  TORCH_CHECK(n >= 0, "n must be non-negative, got", n);
  TORCH_CHECK(!generator.has_value() || (generator.has_value() && result.device() == generator->device()), "Expected a '", result.device(), "' generator device but found '", generator->device(), "'");
  check_supported_max_int_with_precision(n, result);
  result.resize_({n});

  auto uniform_mps = at::native::empty_mps(result.sizes(), kFloat, c10::nullopt, kMPS);
  uniform_mps = at::uniform(uniform_mps, 0.0, 1.0, generator);

  MPSStream *stream = getCurrentMPSStream();

  struct CachedGraph : public MPSCachedGraph
  {
    CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
    MPSGraphTensor* uniformTensor_ = nil;
    MPSGraphTensor* outputTensor_ = nil;
  };

  MPSGraphCache* cache_ = MPSGraphCache::getInstance();
  @autoreleasepool {
    string key = "randperm_out_mps" + getTensorsStringKey({result}) ;
    CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));

    if(!cachedGraph) {
      MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {

        CachedGraph *newCachedGraph = nil;
        @autoreleasepool {

          MPSGraph* mpsGraph = make_mps_graph();
          newCachedGraph = new CachedGraph(mpsGraph);

          newCachedGraph->uniformTensor_ = mpsGraphRankedPlaceHolder(mpsGraph, uniform_mps);
          newCachedGraph->outputTensor_ = [mpsGraph argSortWithTensor:newCachedGraph->uniformTensor_
                                                                 axis:0
                                                                 name:nil];
        }
        return newCachedGraph;
      });
      cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
    }

    Placeholder uniformPlaceholder = Placeholder(cachedGraph->uniformTensor_, uniform_mps);
    Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, result);

  // Create dictionary of inputs and outputs
    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
      uniformPlaceholder.getMPSGraphTensor() : uniformPlaceholder.getMPSGraphTensorData()
    };

    NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
      outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
    };

    runMPSGraph(stream, cachedGraph->graph(), feeds, results);
 }

  return result;
}

} // namespace native
} // namespace at
