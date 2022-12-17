//  Copyright © 2022 Apple Inc.

#include <ATen/native/CPUFallback.h>

namespace at {

void mps_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack)
{
  TORCH_WARN_ONCE("The operator '", op.schema().operator_name(), "' is not currently supported ",
                  "on the MPS backend and will fall back to run on the CPU.",
                  " This may have performance implications.");
  native::cpu_fallback(op, stack);
}

void mps_error_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack)
{
  TORCH_CHECK_NOT_IMPLEMENTED(false, "The operator '", op.schema().operator_name(), "' is not currently implemented ",
    "for the MPS device. If you want this op to be added in priority during the prototype ",
    "phase of this feature, please comment on https://github.com/pytorch/pytorch/issues/77764. ",
    "As a temporary fix, you can set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` ",
    "to use the CPU as a fallback for this op. WARNING: this will be slower than running natively ",
    "on MPS.")
}


// This dispatch should never be called for tensor on MPS but is frequently called
// If one of them are on CPU
Tensor slow_conv2d_forward_mps(
    const Tensor &self,
    const Tensor &weight,
    IntArrayRef kernel_size,
    const c10::optional<Tensor> &bias,
    IntArrayRef stride,
    IntArrayRef padding) {
   TORCH_CHECK(self.device() == weight.device(), __func__, ": input(device='", self.device(), "') and weight(device=", weight.device(), "')  must be on the same device");
   TORCH_INTERNAL_ASSERT(false, __func__, " should not be called for both tensors on MPS device");
}

TORCH_LIBRARY_IMPL(_, MPS, m) {
  static const char *enable_mps_fallback = getenv("PYTORCH_ENABLE_MPS_FALLBACK");
  if (!enable_mps_fallback || std::stoi(enable_mps_fallback) == 0) {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&mps_error_fallback>());
  } else {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  }
}

TORCH_LIBRARY_IMPL(aten, MPS, m) {
  // These ops are not supported via MPS backend currently, and we fallback to run on CPU.
  // For the rest of unsupported ops the user needs to pass 'PYTORCH_ENABLE_MPS_FALLBACK=1'
  // to fallback on CPU, otherwise we will error out.
  m.impl("bitwise_left_shift.Tensor_out", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("bitwise_right_shift.Tensor_out", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("embedding_renorm_", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("linalg_svd", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("linalg_svd.U", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("repeat_interleave.Tensor", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("repeat_interleave.self_Tensor", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("repeat_interleave.self_int", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("_fft_c2c", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("_fft_r2c", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("linalg_vector_norm", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("_slow_conv2d_forward", slow_conv2d_forward_mps);
  m.impl("upsample_nearest3d.vec", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("topk.values", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("norm.out", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("norm.dtype_out", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("sigmoid.out", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("pow.Tensor_Tensor_out", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("pow.Tensor_Scalar_out", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("sgn.out", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("cumsum.out", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("_cdist_forward", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("nll_loss_forward.output", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("nll_loss_backward.grad_input", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("nll_loss2d_forward.output", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("nll_loss2d_forward", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("nll_loss2d_backward.grad_input", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("nll_loss2d_backward", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("minimum.out", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  //m.impl("_index_put_impl_", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("argmax.out", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("smooth_l1_loss.out", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("smooth_l1_loss_backward.grad_input", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("upsample_nearest2d.out", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("upsample_nearest2d_backward.grad_input", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("ne.Scalar_out", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("ne.Tensor_out", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("cos.out", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("sigmoid_backward.grad_input", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("min", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("min.dim_min", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("upsample_bilinear2d.out", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("upsample_bilinear2d_backward.grad_input", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("max", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("max.dim_max", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  //m.impl("index.Tensor_out", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("abs.out", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("exp.out", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("sin.out", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("log.out", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("div.out_mode", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("div.out", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("ge.Scalar_out", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
  m.impl("ge.Tensor_out", torch::CppFunction::makeFromBoxedFunction<&mps_fallback>());
}

} // namespace at
