#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_fused_sdp_choice_native.h>
#include <ATen/ops/_scaled_dot_product_attention_mps.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>
#include <ATen/ops/zeros_like.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/ones.h>
#include <ATen/ops/ones_like_native.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/MPSGraphSonomaOps.h>
#include <ATen/mps/MPSProfiler.h>

#endif

#include<iostream>
namespace at {
namespace native {

struct CachedGraph : public mps::MPSCachedGraph {
  CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
  std::vector<MPSGraphTensor*> inputTensors_;
  MPSGraphTensor* outputTensor_ = nil;
};


//Need a right way to do this definition, now its just here to prevent compilation error.
Tensor _fused_scaled_dot_product_attention_mps(Tensor const& query, Tensor const& key, Tensor const& value, double dropout_p, bool is_causal, c10::optional<double> scale);

//Do I need to do this in MPS namespace? Directly it causes an issue so it needs to be reflected elsewhere.
Tensor _fused_scaled_dot_product_attention_mps(Tensor const& query, Tensor const& key, Tensor const& value, double dropout_p, bool is_causal, c10::optional<double> scale) {
    using namespace mps;
    if (query.numel() == 0 || key.numel() == 0 || value.numel() == 0) {
      //TODO: Check if zeros is the expectation in this case. Or just empty.
      return at::zeros_like(query);                                                                      
    }          

    double scale_;
    if(scale) {
        scale_ = scale.value();
    } else {
        scale_ = 1.0 / sqrt(query.size(-1));
    }

    const int64_t batch_size = query.size(0);
    const int64_t num_heads = query.size(1);
    const int64_t max_seqlen_batch_q = query.size(2);
    const int64_t head_dim = query.size(3);
    
    const int64_t max_seqlen_batch_k = key.size(2);
    const int64_t max_seqlen_batch_v = value.size(2);

    Tensor out = at::zeros_like(query, query.options());
    const auto L = query.size(-2), S = key.size(-2);
    auto mask = at::zeros({L, S}, query.options());
    if (is_causal) {
        auto temp = at::ones({L, S}, query.options().dtype(at::kBool)).tril();
        mask.masked_fill_(temp.logical_not(), -std::numeric_limits<double>::infinity());
    }


    MPSStream* stream = getCurrentMPSStream();
    @autoreleasepool {
      string cacheKey = "fused_sdpa_" + getTensorsStringKey({query}); // + std::to_string(scale);
      auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(cacheKey, [&](auto mpsGraph, auto newCachedGraph) {
        auto mpsDtype = getMPSDataType(query);
        MPSGraphTensor* queryTensor = mpsGraphUnrankedPlaceHolder(mpsGraph, mpsDtype);
        MPSGraphTensor* keyTensor = mpsGraphUnrankedPlaceHolder(mpsGraph, mpsDtype);
        MPSGraphTensor* valueTensor = mpsGraphUnrankedPlaceHolder(mpsGraph, mpsDtype);
        MPSGraphTensor* maskTensor = mpsGraphUnrankedPlaceHolder(mpsGraph, mpsDtype);
        newCachedGraph->inputTensors_ = {queryTensor, keyTensor, valueTensor, maskTensor};

        MPSGraphTensor *sdpa = [mpsGraph scaledDotProductAttentionWithQueryTensor:queryTensor
                                                   keyTensor:keyTensor
                                                 valueTensor:valueTensor
                                                  maskTensor:maskTensor
                                                       scale:scale_
                                                        name:nil];


        newCachedGraph->outputTensor_ = sdpa;
        return newCachedGraph;
      });

      Placeholder queryPlaceholder = Placeholder(cachedGraph->inputTensors_[0], query, getMPSShape(query));
      Placeholder keyPlaceholder = Placeholder(cachedGraph->inputTensors_[1], key, getMPSShape(key));
      Placeholder valuePlaceholder = Placeholder(cachedGraph->inputTensors_[2], value, getMPSShape(value));
      Placeholder maskPlaceholder = Placeholder(cachedGraph->inputTensors_[3], mask, getMPSShape(mask));
      Placeholder outputPlaceholder = Placeholder(cachedGraph->outputTensor_, out);
      auto feeds = dictionaryFromPlaceholders(queryPlaceholder, keyPlaceholder, valuePlaceholder, maskPlaceholder);

      runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, outputPlaceholder);
    }
    return out;
}


int64_t _fused_sdp_choice_mps(const Tensor& query_, const Tensor& key, const Tensor& value, const std::optional<Tensor>& attn_mask_, double dropout_p, bool is_causal, c10::optional<double> scale){
  sdp::sdp_params kernel_params{query_, key, value, attn_mask_, dropout_p, is_causal};
  auto backend = sdp::SDPBackend::mps_attention; //select_sdp_backend(kernel_params);
  if (backend == sdp::SDPBackend::error) {
    TORCH_CHECK(
        false,
        "No viable backend for scaled_dot_product_attention was found. ",
        "This is likely due to turning off both the math kernel and the fused kernels.");
  }
  return static_cast<int64_t>(backend);
}


REGISTER_MPS_DISPATCH(_fused_sdp_choice_stub, &_fused_sdp_choice_mps);
}} // namespace at::native
