//  Copyright © 2022 Apple Inc.

#include <ATen/ATen.h>
#include <ATen/Tensor.h>
#include <ATen/Utils.h>
#include <ATen/TensorUtils.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/Pool.h>
#include <torch/library.h>

namespace at {
namespace native {

TORCH_IMPL_FUNC(adaptive_max_pool2d_out_mps)
  (const Tensor& input,
   IntArrayRef output_size,
   const Tensor& output,
   const Tensor& indices) {

  for (int64_t i = 1; i < input.ndimension(); i++) {
    TORCH_CHECK(input.size(i) > 0,
      "adaptive_max_pool2d(): Expected input to have non-zero size for non-batch dimensions, "
      "but input has sizes ", input.sizes(), " with dimension ", i, " being "
      "empty");
  }

  int64_t isizeH = input.size(-2);
  int64_t isizeW = input.size(-1);

  int64_t osizeH = output_size[0];
  int64_t osizeW = output_size[1];

  if(input.suggest_memory_format() == at::MemoryFormat::ChannelsLast)
    TORCH_CHECK(input.ndimension() == 4,
                    "adaptive_avg_pool2d(): Expected 4D tensor, but got ",
                    input.sizes())

  switch (input.suggest_memory_format()) {
    case at::MemoryFormat::Contiguous:
    case at::MemoryFormat::ChannelsLast:
      break;
    default:
        TORCH_CHECK(
          false,
          "Unsupported memory format. Supports only ChannelsLast, Contiguous")
  }

  int64_t strideH;
  int64_t strideW;
  int64_t kernel_sizeH;
  int64_t kernel_sizeW;

  mps::set_kernel_params(isizeH, isizeW,
                         osizeH, osizeW,
                         strideH, strideW,
                         kernel_sizeH, kernel_sizeW);

  auto outputs = at::max_pool2d_with_indices(input,
                              IntArrayRef({kernel_sizeH, kernel_sizeW}),
                              IntArrayRef({strideH, strideW}),
                              IntArrayRef({0, 0}),
                              IntArrayRef({1, 1}),
                              false);

  output.copy_(std::get<0>(outputs));
  indices.copy_(std::get<1>(outputs));
}

TORCH_IMPL_FUNC(adaptive_max_pool2d_backward_out_mps)
  (const Tensor& gradOutput,
   const Tensor& input,
   const Tensor& indices,
   const Tensor& gradInput) {

  int64_t isizeH = input.size(-2);
  int64_t isizeW = input.size(-1);
  int64_t osizeH = gradOutput.size(-2);
  int64_t osizeW = gradOutput.size(-1);

  int64_t strideH, strideW, kernel_sizeH, kernel_sizeW;

  mps::set_kernel_params(isizeH, isizeW,
                         osizeH, osizeW,
                         strideH, strideW,
                         kernel_sizeH, kernel_sizeW);

  auto returnGradInput = at::max_pool2d_with_indices_backward(gradOutput,
                                                              input,
                                                              IntArrayRef({kernel_sizeH, kernel_sizeW}),
                                                              IntArrayRef({strideH, strideW}),
                                                              IntArrayRef({0, 0}),
                                                              IntArrayRef({1, 1}),
                                                              false,
                                                              indices);

  gradInput.copy_(returnGradInput);

}

}
}
