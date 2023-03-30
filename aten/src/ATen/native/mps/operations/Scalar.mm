//  Copyright © 2022 Apple Inc.

#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/mps/Copy.h>

namespace at::native {

Scalar _local_scalar_dense_mps(const Tensor& self) {
  Scalar r;

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
    at::ScalarType::Half, at::ScalarType::Bool, at::ScalarType::BFloat16, self.scalar_type(), "_local_scalar_dense_mps", [&] {
      Tensor cpu_output = at::empty({1}, TensorOptions(at::CPU(self.scalar_type())));
      mps::mps_copy_(cpu_output, self, false);
      scalar_t cpu_scalar = *cpu_output.data_ptr<scalar_t>();
      r = Scalar(cpu_scalar);
   });

  return r;
}

} // namespace at::native
