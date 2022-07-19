//  Copyright © 2023 Apple Inc.
#pragma once

#include <ATen/native/mps/OperationUtils.h>

namespace at::native::mps {

enum class ScalarOpCategories {
    BINARY_OPS,
    BITWISE_OPS,
    // TODO: add support for Unary Ops
};

bool scalar_ops_mps(const std::string& op,
                    const Tensor& self,
                    const Tensor& other,
                    const Tensor& output,
                    ScalarOpCategories scalarOpCategory);

}
