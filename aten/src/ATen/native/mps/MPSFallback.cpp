//  Copyright © 2022 Apple Inc.

#include <ATen/ATen.h>
#include <ATen/native/mps/Copy.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/MathBitsFallback.h>
#include <ATen/core/ivalue.h>
#include <ATen/core/stack.h>
#include <ATen/core/boxing/KernelFunction.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <ATen/Functions.h>

namespace at {


// TODO: This is based on the CPUFallback. This is very much work in progress.
// Changes we made on top of CPUFallback
// 1. Instead of using trgt_device computation, kMPS to set the device
// 2. Use mps_copy for the Blits
// 
// We are still fixing cases and we will refactor this along with CPUFallback so
// that we have a unified Fallback solution which works across devices.
struct MPSFallback {
  MPSFallback() : key(DispatchKey::MPS), op_name("mps") {}
  std::vector<at::Tensor> to_cpu(const at::TensorList& tensors) {
      // We can't just call at::to_cpu() on the entire list of Tensors
      // Because it will break on undefined tensors. Separate out undefined tensors first.
      std::vector<at::Tensor> cpu_tensors(tensors.size());
      std::vector<at::Tensor> valid_tensors;
      std::vector<bool> to_translate(tensors.size());
      for (const auto i : c10::irange(tensors.size())) {
          const at::Tensor& tensor = tensors[i];
          // Explicitly handling undefined tensors here.
          // TODO: Move this logic in mps_copy to handle the at::_to_cpu.
          if (tensor.defined()) {
              to_translate[i] = true;
              valid_tensors.push_back(tensor);
          } else {
              cpu_tensors[i] = tensor;
          }
      }
      auto cpu_valid_tensors = at::_to_cpu(valid_tensors);
      for (size_t i = 0, defined_pos = 0; i < tensors.size(); ++i) {
          if (to_translate[i]) {
              cpu_tensors[i] = std::move(cpu_valid_tensors[defined_pos++]);
          }
      }
    return cpu_tensors;
  }

  void fallback_impl(const c10::OperatorHandle& op, DispatchKeySet dispatch_keys, torch::jit::Stack* stack) {
    auto& schema_args = op.schema().arguments();
    const auto num_arguments = schema_args.size();
    auto arguments = torch::jit::last(stack, num_arguments);
    const auto arguments_begin = stack->size() - num_arguments;

    std::vector<at::Tensor> tensor_args;
    std::vector<int> tensor_args_indices;

    std::vector<c10::List<at::Tensor>> tensorlist_args;

    // Step 1: Convert all non-CPU tensor inputs into CPU tensors
    // and put them on the stack at the correct indices.
    for (const auto idx : c10::irange(arguments.size())) {
      const auto& ivalue = arguments[idx];
      if (ivalue.isTensor()) {
        tensor_args.push_back(ivalue.toTensor());
        tensor_args_indices.push_back(idx);
      } else if (ivalue.isTensorList()) {
        auto cpu_ivalue = c10::IValue(c10::List<at::Tensor>(to_cpu(ivalue.toTensorList().vec())));
        (*stack)[arguments_begin + idx] = std::move(cpu_ivalue);
        tensorlist_args.push_back(ivalue.toTensorList());
      }
    }
    auto cpu_tensors = to_cpu(tensor_args);

    for (const auto i : c10::irange(tensor_args_indices.size())) {
      auto idx = tensor_args_indices[i];
      (*stack)[arguments_begin + idx] = c10::IValue(cpu_tensors[i]);
    }

    // Step 2: Call the underlying CPU implementation of the operator
    op.redispatchBoxed(c10::DispatchKeySet(c10::DispatchKey::CPU), stack);

    // Step 3: We need to take special care to handle mutable aliases properly:
    // If any input tensors are mutable aliases, we need to
    // directly copy the updated data on the CPU tensors back to the original inputs.
    for (const auto i : c10::irange(tensor_args_indices.size())) {
      auto tensor_idx = tensor_args_indices[i];
      const AliasInfo* alias_info = schema_args[tensor_idx].alias_info();
      if (alias_info != nullptr && alias_info->isWrite()) {
        at::native::mps::mps_copy_(tensor_args[i], cpu_tensors[i], false);
      }
    }

    // Step 4: Convert any CPU output tensors back to the original input device.
    const auto& schema_returns = op.schema().returns();
    const auto& num_returns = schema_returns.size();
    auto returns = torch::jit::last(stack, num_returns);
    const auto returns_begin = stack->size() - num_returns;

    for (const auto idx : c10::irange(returns.size())) {
      if (returns[idx].isTensor()) {
        const auto& return_tens = returns[idx].toTensor();
        if (return_tens.defined()) {
          const AliasInfo* alias_info = schema_returns[idx].alias_info();
          if (alias_info != nullptr && alias_info->isWrite()) {
            bool found_alias = false;
            for (const auto i : c10::irange(tensor_args_indices.size())) {
              auto input_tensor_idx = tensor_args_indices[i];
              const auto& input_tensor = cpu_tensors[i];
              const AliasInfo* input_alias_info = schema_args[input_tensor_idx].alias_info();
              if (input_tensor.defined() && (alias_info == input_alias_info || (input_alias_info != nullptr && *alias_info == *input_alias_info))) {
                (*stack)[returns_begin + idx] = c10::IValue(tensor_args[i]);
                found_alias = true;
                break;
              }
            }
            TORCH_CHECK(found_alias, "The operator ", op.schema().operator_name(), " appears to have invalid alias information. ",
                        "Found a return tensor argument with a mismatched mutable alias: ", schema_returns[idx]);
          } else {
            // Set the target device to MPS.
            c10::optional<c10::Device> tgt_device = c10::Device(kMPS);
            if (alias_info != nullptr && !alias_info->isWrite()) {
              std::stringstream dev_str;
              if (tgt_device) {
                  dev_str << *tgt_device;
              } else {
                  dev_str << "<none>";
              }
              TORCH_WARN(false, "The operator ", op.schema().operator_name(), " appears to be a view operator, ",
                          "but it has no implementation for the backend \"", dev_str.str(), "\". View operators don't support ",
                          "falling back to run on the CPU, since the tensor's storage cannot be shared across devices.");
            }
            // Case (2): copy case. Copy the cpu output tensor to the original device.
            if (tgt_device) {
                (*stack)[returns_begin + idx] = c10::IValue(returns[idx].toTensor().to(*tgt_device));
            }
          }
        }
      }
    }
  }

  DispatchKey key;
  string op_name;
};

void mpsFallback(const c10::OperatorHandle& op, DispatchKeySet dispatch_keys, torch::jit::Stack* stack) {
  MPSFallback object;
  object.fallback_impl(op, dispatch_keys, stack);
}

TORCH_LIBRARY_IMPL(_, MPS, m) {
  static const char *mps_fallback = getenv("PYTORCH_DISABLE_MPS_FALLBACK");
  if(mps_fallback != NULL && std::stoi(mps_fallback) == 1) {
    return;
  }
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&mpsFallback>());
}

} // namespace at
