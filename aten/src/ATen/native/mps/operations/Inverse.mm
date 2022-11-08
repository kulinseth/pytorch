#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/Utils.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>
#include <torch/library.h>
#include <c10/util/Optional.h>


namespace at {
namespace native {

TORCH_IMPL_FUNC(linalg_inv_ex_out_mps)(const Tensor& A, bool check_errors, const Tensor& result, const Tensor& info)
{
    TORCH_CHECK(result.is_mps(), "Output tensor is not MPS");

    using namespace mps;
    MPSStream* stream = getCurrentMPSStream();

    struct CachedGraph : public MPSCachedGraph
    {
        CachedGraph(MPSGraph *graph) : MPSCachedGraph(graph) {}
        MPSGraphTensor* inputTensor_ = nil;
        MPSGraphTensor* outputTensor_ = nil;
    };

    MPSGraphCache* cache_ = MPSGraphCache::getInstance();

    @autoreleasepool {
        string key = "inv_out_mps" + getTensorsStringKey({A});
        CachedGraph* cachedGraph = static_cast<CachedGraph *>(cache_->LookUp(key));
        if(!cachedGraph)
        {
            MPSCachedGraph *tmpCachedGraph = cache_->CreateCachedGraph(key, ^ MPSCachedGraph * () {

                CachedGraph *newCachedGraph = nil;
                
                @autoreleasepool {
                    MPSGraph* mpsGraph = make_mps_graph();
                    newCachedGraph = new CachedGraph(mpsGraph);
                    MPSGraphTensor* inputTensor= mpsGraphRankedPlaceHolder(mpsGraph, A);
                    MPSGraphTensor* outputTensor = [mpsGraph inverseOfTensor: inputTensor
                                                                    name: nil];

                    newCachedGraph->inputTensor_ = inputTensor;
                    newCachedGraph->outputTensor_ = outputTensor;
                }

                return newCachedGraph;

            });
            cachedGraph = static_cast<CachedGraph *>(tmpCachedGraph);
        }

        Placeholder inputPlaceholder = Placeholder(cachedGraph->inputTensor_, A);
        Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, output);

        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds = @{
            inputPlaceholder.getMPSGraphTensor() : inputPlaceholder.getMPSGraphTensorData()
        };

        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results = @{
            outputPlaceholder.getMPSGraphTensor() : outputPlaceholder.getMPSGraphTensorData()
        };

        runMPSGraph(stream, cachedGraph->graph(), feeds, results);
        result.copy_(output);

    }
}
}
}