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
  virtual ~MPSHooksInterface() {}

  // Initialize the MPS library state
  virtual void initMPS() const {
    AT_ERROR("Cannot initialize MPS without MPS backend.");
  }

  virtual bool hasMPS() const {
    return false;
  }

  virtual const Generator& getDefaultMPSGenerator(DeviceIndex device_index = -1) const {
    (void)device_index; // Suppress unused variable warning
    AT_ERROR("Cannot get default MPS generator without MPS backend.");
  }

  virtual Allocator* getMPSDeviceAllocator() const {
    AT_ERROR("MPSDeviceAllocator requires MPS.");
  }
};

struct TORCH_API MPSHooksArgs {};

C10_DECLARE_REGISTRY(MPSHooksRegistry, MPSHooksInterface, MPSHooksArgs);
#define REGISTER_MPS_HOOKS(clsname) \
  C10_REGISTER_CLASS(MPSHooksRegistry, clsname, clsname)

namespace detail {
TORCH_API const MPSHooksInterface& getMPSHooks();

} // namespace detail
} // namespace at
