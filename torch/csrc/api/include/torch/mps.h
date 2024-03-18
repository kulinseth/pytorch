#pragma once

#include <torch/csrc/Export.h>

#include <cstddef>
#include <cstdint>

#ifdef __OBJC__
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
typedef id<MTLCommandBuffer> MTLCommandBuffer_t;
#else
typedef void* MTLCommandBuffer_t;
typedef void* MTLCommandBuffer;
typedef void* dispatch_queue_t;
#endif

namespace torch {
namespace mps {

/// Returns true if MPS device is available.
bool TORCH_API is_available();

/// Sets the RNG seed for the MPS device.
void TORCH_API manual_seed(uint64_t seed);

/// Waits for all streams on a MPS device to complete.
void TORCH_API synchronize();

/// Submits the command buffer to run on the GPU
void TORCH_API commit();

/// Get the current command buffer to encode the Metal commands
MTLCommandBuffer_t TORCH_API get_command_buffer();

/// Get the dispatch_queue_t to synchronize encoding the custom kernels
/// with the PyTorch MPS backend
dispatch_queue_t TORCH_API get_dispatch_queue();

} // namespace mps
} // namespace torch
