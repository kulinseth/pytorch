//  Copyright © 2022 Apple Inc.

#pragma once

#include <ATen/detail/MPSHooksInterface.h>
#include <ATen/Generator.h>
//#include <ATen/mps/MPSStream.h>
#include <c10/util/Optional.h>

namespace at { namespace mps {

// The real implementation of MPSHooksInterface
struct MPSHooks : public at::MPSHooksInterface {
  MPSHooks(at::MPSHooksArgs) {}
  void initMPS() const override;
  bool hasMPS() const override;
  bool isOnMacOS13orNewer(unsigned minor) const override;
  Allocator* getMPSDeviceAllocator() const override;
  const Generator& getDefaultMPSGenerator() const override;
  void deviceSynchronize() const override;
  void emptyCache() const override;
  size_t getCurrentAllocatedMemory() const override;
  size_t getDriverAllocatedMemory() const override;
  void setMemoryFraction(double ratio) const override;
  int getDevice() const override;
  void setDevice(int d) const override;
  //MPSStream getCurrentMPSStream() const override;
  //MPSStream getDefaultMPSStream() const override;
};

}} // at::mps
