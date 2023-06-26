//  Copyright © 2022 Apple Inc.

#pragma once

#include <c10/core/Allocator.h>
#include <ATen/core/Generator.h>
#include <c10/util/Exception.h>
#include <c10/util/Registry.h>

#include <cstddef>
#include <functional>

namespace at {
class Context;
}

namespace at {


struct TORCH_API MPSHooksInterface {
  // this fails the implementation if MPSHooks functions are called, but
  // MPS backend is not present.
  #define FAIL_MPSHOOKS_FUNC(func) \
    AT_ERROR("Cannot execute ", func ,"() without MPS backend.");

  virtual ~MPSHooksInterface() = default;

  // Initialize the MPS library state
  virtual void initMPS() const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual bool hasMPS() const {
    return false;
  }
  virtual bool isOnMacOS13orNewer(unsigned minor = 0) const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual const Generator& getDefaultMPSGenerator() const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual Allocator* getMPSDeviceAllocator() const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual void synchronizeStream() const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual void commitStream() const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual void* getCommandBuffer() const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual void* getDispatchQueue() const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual void emptyCache() const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual size_t getCurrentAllocatedMemory() const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual size_t getDriverAllocatedMemory() const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual void setMemoryFraction(double /*ratio*/) const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual void setAllocatorSettings(const std::string& configStr) const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual void profilerStartTrace(const string& mode, bool waitUntilCompleted) const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual void profilerStopTrace() const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual id_t acquireEvent(bool enable_timing) const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual void releaseEvent(id_t event_id) const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual void recordEvent(id_t event_id) const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual void waitForEvent(id_t event_id) const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual void synchronizeEvent(id_t event_id) const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual bool queryEvent(id_t event_id) const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
  virtual double elapsedTimeOfEvents(id_t start_event_id, id_t end_event_id) const {
    FAIL_MPSHOOKS_FUNC(__func__);
  }
};

struct TORCH_API MPSHooksArgs {};

TORCH_DECLARE_REGISTRY(MPSHooksRegistry, MPSHooksInterface, MPSHooksArgs);
#define REGISTER_MPS_HOOKS(clsname) \
  C10_REGISTER_CLASS(MPSHooksRegistry, clsname, clsname)

namespace detail {
TORCH_API const MPSHooksInterface& getMPSHooks();

} // namespace detail
} // namespace at
