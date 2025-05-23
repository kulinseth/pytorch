//  Copyright © 2023 Apple Inc.

#pragma once

#include <c10/core/Allocator.h>
#include <c10/util/Registry.h>
#include <ATen/core/ATen_fwd.h>

#define MB(x) (x * 1048576UL)

namespace at { namespace mps {

// this is a public interface to access MPSAllocator.
// Do not declare methods that would depend on MPS or Metal frameworks.
class IMPSAllocator : public c10::Allocator {
public:
  // see the comments in MPSAllocator.h for the description of these methods.
  virtual void emptyCache() const = 0;
  virtual void freeInactiveBuffers() const = 0;
  virtual ssize_t getUnalignedBufferSize(void* ptr) const = 0;
  virtual IntArrayRef getBufferShape(void* ptr) const = 0;
  virtual id_t getBufferId(void* ptr) const = 0;
  virtual void setBufferShape(void* ptr, const IntArrayRef& shape) const = 0;
  virtual bool isSharedBuffer(void* ptr) const = 0;
  virtual bool isSharedStorageSupported() const = 0;
  virtual c10::DataPtr allocScalarBufferWithValue(void* value, size_t size) const = 0;
  virtual std::string formatSize(size_t size) const = 0;
  virtual void setHighWatermarkRatio(double ratio) const = 0;
  virtual ssize_t getLowWatermarkValue() const = 0;
  virtual size_t getTotalAllocatedMemory() const = 0;
  virtual size_t getCurrentAllocatedMemory() const = 0;
  virtual size_t getDriverAllocatedMemory() const = 0;
  virtual void setAllocatorSettings(const std::string& configStr) const = 0;
  virtual std::pair<void*, uint32_t> getSharedBufferPtr(void* buffer) const = 0;
  virtual bool recordEvents(c10::ArrayRef<void*> buffers) const = 0;
  virtual bool waitForEvents(c10::ArrayRef<void*> buffers) const = 0;
};

class IMpsAllocatorCallback {
 public:
  enum class EventType {
    ALLOCATED, // buffer got allocated to be used immediately
    RECYCLED,  // buffer pulled from free list to be reused
    FREED,     // buffer put to free list for future recycling
    RELEASED,  // buffer memory released
    ALLOCATION_FAILED // buffer allocation failed
  };
  virtual ~IMpsAllocatorCallback() = default;
  virtual void executeMPSAllocatorCallback(void* ptr, EventType event) = 0;
};

// MPS allocator will execute every registered callback when a block of memory is freed.
C10_DECLARE_REGISTRY(MPSAllocatorCallbacksRegistry, IMpsAllocatorCallback);
#define REGISTER_MPS_ALLOCATOR_CALLBACK(name, ...) \
  C10_REGISTER_CLASS(MPSAllocatorCallbacksRegistry, name, __VA_ARGS__);

IMPSAllocator* getIMPSAllocator(bool sharedAllocator = false);

}} // namespace at::mps
