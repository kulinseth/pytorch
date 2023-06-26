//  Copyright © 2023 Apple Inc.

#pragma once

#include <ATen/mps/MPSAllocatorInterface.h>
#include <cstdio>
#include <iostream>

namespace at::mps {
namespace HeapAllocator {

// largest "small" allocation is 1 MiB
static const size_t kMaxSmallAlloc = MB(1);
// allocations between 1 and 10 MiB may use kLargeHeap
static const size_t kMinLargeAlloc = MB(10);
// round up large allocations to 2 MiB
static const size_t kRoundLarge = MB(2);
// Min heap size to pack "small" allocations
static const size_t kMinSmallHeap = MB(8);
// Min heap size to pack "large" allocations
static const size_t kMinLargeHeap = MB(32);
// Min heap size to pack "extra large" allocations
static const size_t kMinXLargeHeap = MB(1024);
// largest "scalar" allocation
static const size_t kMaxScalarAlloc = (sizeof(int64_t));
// smallest size that gets round up to the next power of 2
static const size_t kMinRoundUpSize = 1024;

// debug verbosity flags
enum DebugVerbosity : uint32_t {
  SILENT      = 0,
  PROFILING   = (1 << 0), // print generic profiling data for total system memory usage
  ALLOCATIONS = (1 << 1), // print buffer allocations
  RECYCLES    = (1 << 2), // print buffer recycling
  RELEASES    = (1 << 3), // print buffer releases
  LARGE_ONLY  = (1 << 4), // only log large buffer pool transactions
  VERBOSITY_COUNT = (LARGE_ONLY << 1) - 1,
};

class MPSAllocatorConfig {
public:
  explicit MPSAllocatorConfig(bool hasUnifiedMemory, size_t maxDeviceSize) :
      m_has_unified_memory(hasUnifiedMemory), m_max_device_size(maxDeviceSize) {
    // config str is either read from this env-var or via set_allocator_settings() API
    const char* configs_env_str = getenv("PYTORCH_MPS_ALLOC_CONF");
    parseArgs(configs_env_str);

    // explicitly initialize configs to enable printing debug values
    if (m_high_watermark_ratio < 0.0) {
      setHighWatermarkRatio(default_high_watermark_ratio);
    }
    if (m_low_watermark_ratio < 0.0) {
      setLowWatermarkRatio(m_has_unified_memory ? default_low_watermark_ratio_unified :
                                                  default_low_watermark_ratio_discrete);
    }
    if (m_xlarge_heap_divisor == std::numeric_limits<size_t>::max()) {
      setXLargeHeapDivisor(0); // 0 initializes to default value
    }
    if (m_large_heap_divisor == std::numeric_limits<size_t>::max()) {
      setLargeHeapDivisor(0); // 0 initializes to default value
    }
    if (m_small_heap_divisor == std::numeric_limits<size_t>::max()) {
      setSmallHeapDivisor(0); // 0 initializes to default value
    }
    if (m_max_pow2_roundup_size_mb == std::numeric_limits<size_t>::max()) {
      setMaxPow2RoundupSize(0); // 0 disables Pow2 Roundup
    }
  }

  // parser based on CUDACachingAllocator
  void parseArgs(const char* env) {
    if (env == nullptr) {
      return;
    }
    std::vector<std::string> config;
    std::vector<char> buf;
    size_t env_length = strlen(env);
    for (size_t i = 0; i < env_length; i++) {
      if (env[i] == ',' || env[i] == ':' || env[i] == '[' || env[i] == ']') {
        if (!buf.empty()) {
          config.emplace_back(buf.begin(), buf.end());
          buf.clear();
        }
        config.emplace_back(1, env[i]);
      } else if (env[i] != ' ') {
        buf.emplace_back(static_cast<char>(env[i]));
      }
    }
    if (!buf.empty()) {
      config.emplace_back(buf.begin(), buf.end());
    }

    for (size_t i = 0; i < config.size(); i++) {
      const std::string& config_str = config[i];
      checkToken(config, ++i, ':');
      TORCH_CHECK(++i < config.size(), "expecting value for the MPSAllocator config '", config_str, "'");
      const std::string& value_str = config[i];

      if (config_str.compare("high_watermark_ratio") == 0) {
        setHighWatermarkRatio(std::stod(value_str));
      } else if (config_str.compare("low_watermark_ratio") == 0) {
        setLowWatermarkRatio(std::stod(value_str));
      } else if (config_str.compare("debug_verbosity") == 0) {
        setDebugVerbosity(std::stol(value_str));
      } else if (config_str.compare("small_heap_divisor") == 0) {
        setSmallHeapDivisor(std::stol(value_str));
      } else if (config_str.compare("large_heap_divisor") == 0) {
        setLargeHeapDivisor(std::stol(value_str));
      } else if (config_str.compare("xlarge_heap_divisor") == 0) {
        setXLargeHeapDivisor(std::stol(value_str));
      } else if (config_str.compare("max_pow2_roundup_size_mb") == 0) {
        setMaxPow2RoundupSize(MB(std::stol(value_str)));
      } else {
        TORCH_CHECK(false, "Unrecognized MPSAllocator Config: '", config_str, "'");
      }
      if (i + 1 < config.size()) {
        checkToken(config, ++i, ',');
      }
    }
  }

  // this is public since it's being used in torch.mps.set_per_process_memory_fraction()
  void setHighWatermarkRatio(double ratio) {
    if (m_high_watermark_ratio != ratio) {
      TORCH_CHECK(ratio >= 0.0 && ratio <= default_high_watermark_upper_bound, "invalid high watermark ratio ", ratio);
      m_high_watermark_ratio = ratio;
      printConfigDebug("High watermark memory allocation limit: ",
                 ratio == 0.0 ? "unlimited" : format_size(highWatermarkLimit()));
    }
  }

  // (see enum DebugVerbosity for description)
  uint32_t debugVerbosity() const {
    return m_debug_verbosity;
  }
  // low watermark size limit (in Bytes) at the time we initialize the allocator
  size_t lowWatermarkLimit() const {
    return m_low_watermark_ratio == 0.0 ? std::numeric_limits<size_t>::max() :
           static_cast<size_t>(m_low_watermark_ratio * (double)m_max_device_size);
  }
  // maximum total size allowed to be allocated
  size_t highWatermarkLimit() const {
    return m_high_watermark_ratio == 0.0 ? std::numeric_limits<size_t>::max() :
           static_cast<size_t>(m_high_watermark_ratio * (double)m_max_device_size);
  }
  // heap size to pack "small" allocations
  size_t smallHeapSize() const {
    return m_small_heap_divisor == 0 ? kMinSmallHeap :
           std::max(kMinSmallHeap, m_max_device_size / m_small_heap_divisor);
  }
  // heap size to pack "large" allocations
  size_t largeHeapSize() const {
    return m_large_heap_divisor == 0 ? kMinLargeHeap :
           std::max(kMinLargeHeap, m_max_device_size / m_large_heap_divisor);
  }
  // heap size to pack "extra large" allocations
  size_t xLargeHeapSize() const {
    return m_xlarge_heap_divisor == 0 ? kMinXLargeHeap :
           std::max(kMinXLargeHeap, m_max_device_size / m_xlarge_heap_divisor);
  }
  // (see m_max_pow2_roundup_size_mb for description)
  size_t maxPow2RoundupSize() const {
    return m_max_pow2_roundup_size_mb;
  }

  static std::string format_size(uint64_t size) {
    std::ostringstream os;
    os.precision(2);
    os << std::fixed;
    if (size <= 1024UL) { os << size << " bytes"; }
    else if (size <= 1048576UL) { os << ((float) size / 1024.0) << " KB"; }
    else if (size <= 1073741824UL) { os << ((float) size / 1048576.0) << " MB"; }
    else { os << ((float) size / 1073741824.0) << " GB"; }
    return os.str();
  }

private:
   // (see m_high_watermark_ratio for description)
  constexpr static double default_high_watermark_ratio = 1.7;
  // we set the allowed upper bound to twice the size of recommendedMaxWorkingSetSize.
  constexpr static double default_high_watermark_upper_bound = 2.0;
  // (see m_low_watermark_ratio for description)
  constexpr static double default_low_watermark_ratio_unified  = 1.0;
  constexpr static double default_low_watermark_ratio_discrete = 1.0;

  // if device has unified memory architecture (UMA)
  bool m_has_unified_memory;
  // size from device.recommendedMaxWorkingSetSize which is typically 75% of total system memory.
  size_t m_max_device_size;
  // high watermark ratio is a hard limit for the total allowed allocations
  // 0. : disables high watermark limit (may cause system failure if system-wide OOM occurs)
  // 1. : recommended maximum allocation size (i.e., device.recommendedMaxWorkingSetSize)
  // >1.: allows limits beyond the device.recommendedMaxWorkingSetSize
  // e.g., value 0.95 means we allocate up to 95% of recommended maximum
  // allocation size; beyond that, the allocations would fail with OOM error.
  // ((-1.0 means it'll be initialized in ctor)
  double m_high_watermark_ratio = -1.0;
  // low watermark ratio is a soft limit to attempt limiting memory allocations up to the lower watermark
  // level by garbage collection or committing command buffers more frequently (a.k.a, adaptive commit).
  // Value between 0 to m_high_watermark_ratio (setting 0.0 disables adaptive commit and garbage collection)
  // e.g., value 0.9 means we 'attempt' to limit allocations up to 90% of recommended maximum
  // allocation size.
  // (-1.0 means it'll be initialized in ctor)
  double m_low_watermark_ratio = -1.0;
  // if 0 sets the size of small heap to default value kSmallHeap, otherwise it'll be
  // (recommendedMaxWorkingSetSize / m_small_heap_divisor).
  size_t m_small_heap_divisor = std::numeric_limits<size_t>::max();
  // if 0 sets the size of large heap to default value kLargeHeap, otherwise it'll be
  // (recommendedMaxWorkingSetSize / m_large_heap_divisor).
  size_t m_large_heap_divisor = std::numeric_limits<size_t>::max();
  // if 0 sets the size of extra large heap to default value kXLargeHeap, otherwise it'll be
  // (recommendedMaxWorkingSetSize / m_xlarge_heap_divisor).
  size_t m_xlarge_heap_divisor = std::numeric_limits<size_t>::max();
  // largest size in Mega bytes that gets round up to the next power of 2 (0 disables rounding)
  size_t m_max_pow2_roundup_size_mb = std::numeric_limits<size_t>::max();
  // debug verbosity to log allocator messages
  uint32_t m_debug_verbosity = DebugVerbosity::SILENT;

  void setDebugVerbosity(uint32_t debug_verbosity) {
    if (m_debug_verbosity != debug_verbosity) {
      TORCH_CHECK(m_debug_verbosity <= DebugVerbosity::VERBOSITY_COUNT, "invalid debug_verbosity value: ", debug_verbosity);
      m_debug_verbosity = debug_verbosity;
    }
  }
  void setLowWatermarkRatio(double ratio) {
    if (m_low_watermark_ratio != ratio) {
      // used for comparison with lower_watermark_ratio
      const double high_watermark_ratio = m_high_watermark_ratio <= 0.0 ? default_high_watermark_upper_bound : m_high_watermark_ratio;
      TORCH_CHECK(ratio >= 0.0 && ratio <= high_watermark_ratio, "invalid low_watermark_ratio value: ", ratio,
                  " (must be less than ", m_high_watermark_ratio, ")");
      m_low_watermark_ratio = ratio;
      printConfigDebug("Low watermark memory allocation size limit: ",
                 ratio == 0.0 ? "unlimited" : format_size(lowWatermarkLimit()));
    }
  }
  void setSmallHeapDivisor(size_t divisor) {
    if (m_small_heap_divisor != divisor) {
      TORCH_CHECK(divisor >= 0 && divisor < m_max_device_size, "invalid small_heap_divisor value: ", divisor);
      m_small_heap_divisor = divisor;
      printConfigDebug("Small heap size: ", format_size(smallHeapSize()));
    }
  }
  void setLargeHeapDivisor(size_t divisor) {
    if (m_large_heap_divisor != divisor) {
      TORCH_CHECK(divisor >= 0 && divisor < m_max_device_size, "invalid large_heap_divisor value: ", divisor);
      m_large_heap_divisor = divisor;
      printConfigDebug("Large heap size: ", format_size(largeHeapSize()));
    }
  }
  void setXLargeHeapDivisor(size_t divisor) {
    if (m_xlarge_heap_divisor != divisor) {
      TORCH_CHECK(divisor >= 0 && divisor < m_max_device_size, "invalid xlarge_heap_divisor value: ", divisor);
      m_xlarge_heap_divisor = divisor;
      printConfigDebug("Extra large heap size: ", format_size(xLargeHeapSize()));
    }
  }
  void setMaxPow2RoundupSize(size_t roundUpSize) {
    if (m_max_pow2_roundup_size_mb != roundUpSize) {
      TORCH_CHECK(roundUpSize >= 0 && roundUpSize < m_max_device_size / 2, "invalid max_pow2_roundup_size_mb value: ", roundUpSize);
      m_max_pow2_roundup_size_mb = roundUpSize;
      printConfigDebug("Max Pow2Roundup size: ", roundUpSize == 0 ? "disabled" : format_size(maxPow2RoundupSize()));
    }
  }
  void printConfigDebug(const std::string& msg, const std::string& val) {
    if (m_debug_verbosity >= DebugVerbosity::PROFILING) {
      std::cerr << msg << val << "\n";
    }
  }
  void checkToken(const std::vector<std::string>& config, size_t i, const char c) {
    TORCH_CHECK(i < config.size() && config[i].compare(std::string(1, c)) == 0,
        "Error parsing MPSAllocator settings, expected '", c, "' after '", config[i-1], "'");
  }
}; // class MPSAllocatorConfig

} // namespace HeapAllocator
} // namespace at::mps
