//  Copyright © 2022 Apple Inc.

#include <c10/util/Exception.h>
#include <ATen/mps/MPSProfiler.h>

// these need to be literal strings when passed to os_signpost*()
// function macros; so no LUTs could be used
#define kMPSProfilerSubSystemStr        "PyTorchMPS"
#define kMPSCategoryEventsStr           "Events"
#define kMPSCategoryIntervalsStr        "Intervals"
#define kSignpostNameRunMPSGraphStr     "PyTorchGraphExecution"
#define kSignpostNameRunCustomKernelStr "PyTorchKernelExecution"
#define kSignpostNameBlitCopyStr        "PyTorchTensorCopy"
#define kEVLogProfileInfoStr            "PYTORCH_MPS_LOG_PROFILE_INFO"
#define kEVTraceSignpostsStr            "PYTORCH_MPS_TRACE_SIGNPOSTS"

namespace at::mps {

MPSProfiler::MPSProfiler(): m_os_log_events(nullptr), m_os_log_intervals(nullptr) {
  // see enum LogOptions for the description.
  static const char *log_options_str = getenv(kEVLogProfileInfoStr);
  m_log_options = log_options_str ? strtol(log_options_str, nullptr, 0) : 0;
  // see enums profilerOptions and SignpostTypes for the description.
  static const char *trace_signpost_str = getenv(kEVTraceSignpostsStr);
  uint32_t trace_signposts = trace_signpost_str ? strtol(trace_signpost_str, nullptr, 0) : 0;

  TORCH_CHECK(m_log_options <= LogOptions::LOG_COUNT,
              "invalid log options ", m_log_options, " passed to ", kEVLogProfileInfoStr)
  // lower 16 bits used for options (see enum ProfileOptions)
  m_profile_options |= trace_signposts & 0xFFFF;
  TORCH_CHECK(m_profile_options <= ProfileOptions::OPTIONS_COUNT,
              "invalid profiling options ", trace_signposts, " passed to ", kEVTraceSignpostsStr)
  // upper 16 bits used for signpost types (see enum SignpostTypes)
  m_signpost_types |= trace_signposts & 0xFFFF0000;
  TORCH_CHECK(m_signpost_types <= SignpostTypes::SIGNPOST_COUNT,
              "invalid signpost types ", trace_signposts, " passed to ", kEVTraceSignpostsStr)

  if (m_profile_options & (ProfileOptions::ALL_SIGNPOST_EVENTS |
                           ProfileOptions::ALL_SIGNPOST_INTERVALS)) {
    // enable all signposts types
    m_signpost_types |= (SignpostTypes::RUN_MPSGRAPH |
                         SignpostTypes::RUN_KERNEL |
                         SignpostTypes::BLIT_COPY);

    m_profile_options |= (m_profile_options & ProfileOptions::ALL_SIGNPOST_INTERVALS) ?
                         ProfileOptions::USE_INTERVALS : ProfileOptions::USE_EVENTS;
  }

  if (m_signpost_types != SignpostTypes::SIGNPOST_NONE) {
    // if no signpost options passed, use interval mode by default
    if (!(m_profile_options & (ProfileOptions::USE_EVENTS | ProfileOptions::USE_INTERVALS))) {
      m_profile_options |= ProfileOptions::USE_INTERVALS;
    }

    // technically, it's possible to trace both events and intervals at the same time
    if (m_profile_options & ProfileOptions::USE_EVENTS) {
      m_os_log_events = os_log_create(kMPSProfilerSubSystemStr, kMPSCategoryEventsStr);
      TORCH_CHECK(m_os_log_events, "failed to create OS signpost log for events profiler");
    }
    if (m_profile_options & ProfileOptions::USE_INTERVALS) {
      m_os_log_intervals = os_log_create(kMPSProfilerSubSystemStr, kMPSCategoryIntervalsStr);
      TORCH_CHECK(m_os_log_intervals, "failed to create OS signpost log for intervals profiler");
    }
  }
}

MPSProfiler::~MPSProfiler() {
  // first make sure completion handlers are completed
  getDefaultMPSStream()->synchronize(SyncType::COMMIT_AND_WAIT);

  // logs kernel profiling results when the process ends (if enabled).
  if (m_log_options & LogOptions::KERNEL_PROFILING) {
    logKernelProfiling(stderr);
  }
  m_kernel_info_list.clear();
  m_copy_info_list.clear();

  if (m_os_log_events) {
    os_release(m_os_log_events);
  }
  if (m_os_log_intervals) {
    os_release(m_os_log_intervals);
  }
}

void MPSProfiler::beginProfileKernel(const void* handle, const std::string& strKey, bool isGraph) {
   // only do profiling if graph/kernel execution profiling or logging are enabled
  if (!isGraphProfilingEnabled() && !isKernelProfilingEnabled()) {
    return;
  }

  if (m_kernel_info_list.count(handle) == 0) {
    auto kernelInfo = std::make_unique<Profiler::KernelInfo>(handle, isGraph,
                         isGraph ? ++m_graph_counter : ++m_kernel_counter, strKey);
    m_kernel_info_list.emplace(handle, std::move(kernelInfo));
  }
  auto& kernelInfo = m_kernel_info_list[handle];
  kernelInfo->runCount++;

  SignpostTypes signpostType = isGraph ? SignpostTypes::RUN_MPSGRAPH : SignpostTypes::RUN_KERNEL;
  if ((m_signpost_types & signpostType) && (m_profile_options & ProfileOptions::USE_INTERVALS)) {
    kernelInfo->signpostId = beginSignpostInterval(signpostType, kernelInfo->toString());
  }
}

void MPSProfiler::beginProfileKernel(const void* handle, const std::string& kernelName, const TensorList& tensors) {
  if (isKernelProfilingEnabled()) {
    std::string profilerStrKey = Profiler::KernelInfo::buildKernelString(kernelName, tensors);
    beginProfileKernel(handle, profilerStrKey, false);
  }
}

void MPSProfiler::endProfileKernel(const void* handle) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(m_kernel_info_list.count(handle), "Failed to get kernel information!");
  auto& kernelInfo = m_kernel_info_list[handle];

  SignpostTypes signpostType = kernelInfo->type == Profiler::BaseInfo::Type::GRAPH ?
                               SignpostTypes::RUN_MPSGRAPH : SignpostTypes::RUN_KERNEL;
  addProfilerCompletedHandler(kernelInfo.get(), signpostType, SyncType::COMMIT);
}

void MPSProfiler::beginProfileCopy(const void* srcBuffer, const void* dstBuffer,
                                   const OptionalTensorRef srcTensor,
                                   const OptionalTensorRef dstTensor,
                                   size_t length, bool isNonBlocking) {
  if (!isCopyProfilingEnabled()) {
    return;
  }
  auto copyInfo = std::make_unique<Profiler::CopyInfo>(dstBuffer, length, ++m_copy_counter, isNonBlocking);
  copyInfo->srcStrKey = Profiler::CopyInfo::buildTensorString(srcBuffer, srcTensor);
  copyInfo->dstStrKey = Profiler::CopyInfo::buildTensorString(dstBuffer, dstTensor);
  m_total_copy_size += length;
  if (m_profile_options & ProfileOptions::USE_INTERVALS) {
    copyInfo->signpostId = beginSignpostInterval(SignpostTypes::BLIT_COPY, copyInfo->toString());
  }
  std::lock_guard<std::mutex> lock(m_mutex);
  // this should not happen since we erase the copy info after profiling/logging it.
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(m_copy_info_list.count(dstBuffer) == 0);
  m_copy_info_list.emplace(dstBuffer, std::move(copyInfo));
}

void MPSProfiler::endProfileCopy(const void* handle) {
  Profiler::CopyInfo* copyInfo = nullptr;
  {
    std::lock_guard<std::mutex> lock(m_mutex);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(m_copy_info_list.count(handle), "Failed to get copy information!");
    copyInfo = m_copy_info_list[handle].get();
  }
  addProfilerCompletedHandler(copyInfo, SignpostTypes::BLIT_COPY, SyncType::COMMIT_AND_WAIT);
}

void MPSProfiler::addProfilerCompletedHandler(Profiler::BaseInfo* info, SignpostTypes signpostType, SyncType syncType) {
  const uint64_t signpostId = info->signpostId;
  // signpost ID is used only for interval-based signposts, and must be non-zero
  if (m_profile_options & ProfileOptions::USE_INTERVALS) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(signpostId, "Signpost interval has no identifier!");
  }
  info->signpostId = 0; // reset signpostId for sanity check on next interval

  auto m_stream = getDefaultMPSStream();
  // NOTE: the following block isn't thread-safe, so make sure to capture profiler's info locally
  [m_stream->commandBuffer() addCompletedHandler:^(id<MTLCommandBuffer> cb) {
    CFTimeInterval gpuTime = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
    CFTimeInterval kernelTime = (cb.kernelEndTime - cb.kernelStartTime) * 1000.0;

    std::string infoStr;
    {
      std::lock_guard<std::mutex> lock(m_mutex);
      info->totalGpuTime += gpuTime;
      infoStr = info->toString(gpuTime, kernelTime);
      // the copy info isn't retained in the list
      if (info->type == Profiler::BaseInfo::Type::COPY) {
        m_copy_info_list.erase(info->handle);
      }
    }
    // logging the copy/kernel info is enabled via the env-var defined in kEVLogProfileInfoStr
    if (m_log_options & (LogOptions::BLIT_COPY_INFO | LogOptions::KERNEL_INFO)) {
      fmt::print(stderr, "{}\n", infoStr);
    }

    // NOTE: it is possible to use both interval and event based signposts at the same time, if required
    if ((m_profile_options & ProfileOptions::USE_EVENTS)) {
      emitSignpostEvent(signpostType, infoStr);
    }
    // GPU time for signpost intervals is calculated based on its duration (which ends with completionHandler),
    if ((m_profile_options & ProfileOptions::USE_INTERVALS)) {
      endSignpostInterval(signpostType, signpostId);
    }
  }];

  m_stream->synchronize((m_profile_options & ProfileOptions::WAIT_UNTIL_COMPLETED) ?
                        SyncType::COMMIT_AND_WAIT : syncType);
}

void MPSProfiler::logKernelProfiling(std::FILE* f) const {
  if (m_kernel_info_list.empty()) {
    // this is not an error, but to let the user know that the
    // LogOptions::KERNEL_PROFILING that they passed to EV is not yielding anything.
    fmt::print(f, "There are no graphs or kernels logged for profiling\n");
    return;
  }
  // dump the kernels info into a vector to sort them
  std::vector<Profiler::KernelInfo*> kernelsList;
  std::transform(m_kernel_info_list.begin(), m_kernel_info_list.end(),
                 std::back_inserter(kernelsList),
                 [](auto& kernelInfo){ return kernelInfo.second.get();} );

  // sort based on "Mean GPU time" in descending order
  std::sort(kernelsList.begin(), kernelsList.end(),
            [](const Profiler::KernelInfo* a, const Profiler::KernelInfo* b) {
              return (a->totalGpuTime / (double) a->runCount) > (b->totalGpuTime / (double) b->runCount);
            });
  // print the table of profiling results
  fmt::print(f, "\n{:-^200}\n{:^6}|{:^7}|{:^15}|{:^16}| {}\n{:-^200}\n",
             fmt::format(" MPS Kernel Profiling: {} graphs, {} kernels, {} copies ({}) ",
                         m_graph_counter, m_kernel_counter, m_copy_counter,
                         getIMPSAllocator()->formatSize(m_total_copy_size)),
             "ID", "#Runs",  "Mean GPU (ms)", "Total GPU (ms)", "Kernel Name", "");

  for (const auto& kernelInfo : kernelsList) {
    fmt::print(f, "{:^7}{:^8}{:^16}{:^17} {}\n",
               fmt::format("{}{}", kernelInfo->type == Profiler::BaseInfo::Type::GRAPH ? "G": "K", kernelInfo->id),
               kernelInfo->runCount,
               fmt::format("{:.4f}", kernelInfo->totalGpuTime / (double) kernelInfo->runCount),
               fmt::format("{:.4f}", kernelInfo->totalGpuTime),
               kernelInfo->strKey);
  }
}

uint64_t MPSProfiler::emitSignpostEvent(SignpostTypes signpost_type, const std::string& msg_str) const {
  if (!(m_signpost_types & signpost_type) ||
      !m_os_log_events || !os_signpost_enabled(m_os_log_events)) {
    return 0;
  }
  const char *msg = msg_str.c_str();
  os_signpost_id_t signpost_id = os_signpost_id_generate(m_os_log_events);

  // need to use switch-case as the signpost names must be literal strings
  switch (signpost_type) {
    case SignpostTypes::RUN_MPSGRAPH:
      os_signpost_event_emit(m_os_log_events, signpost_id, kSignpostNameRunMPSGraphStr, "%s", msg);
      break;
    case SignpostTypes::RUN_KERNEL:
      os_signpost_event_emit(m_os_log_events, signpost_id, kSignpostNameRunCustomKernelStr, "%s", msg);
      break;
    case SignpostTypes::BLIT_COPY:
      os_signpost_event_emit(m_os_log_events, signpost_id, kSignpostNameBlitCopyStr, "%s", msg);
      break;
    default:
      AT_ERROR("unknown SignpostType in MPS profiler");
  }
  return signpost_id;
}

uint64_t MPSProfiler::beginSignpostInterval(SignpostTypes signpost_type, const std::string& msg_str) const {
  if (!(m_signpost_types & signpost_type) ||
      !m_os_log_intervals || !os_signpost_enabled(m_os_log_intervals)) {
    return 0;
  }
  const char *msg = msg_str.c_str();
  os_signpost_id_t signpost_id = os_signpost_id_generate(m_os_log_intervals);

  switch (signpost_type) {
    case SignpostTypes::RUN_MPSGRAPH:
      os_signpost_interval_begin(m_os_log_intervals, signpost_id, kSignpostNameRunMPSGraphStr, "%s", msg);
      break;
    case SignpostTypes::RUN_KERNEL:
      os_signpost_interval_begin(m_os_log_intervals, signpost_id, kSignpostNameRunCustomKernelStr, "%s", msg);
      break;
    case SignpostTypes::BLIT_COPY:
      os_signpost_interval_begin(m_os_log_intervals, signpost_id, kSignpostNameBlitCopyStr, "%s", msg);
      break;
    default:
      AT_ERROR("unknown SignpostType in MPS profiler");
  }
  return signpost_id;
}

void MPSProfiler::endSignpostInterval(SignpostTypes signpost_type, os_signpost_id_t signpost_id) const {
  if (!m_os_log_intervals || !os_signpost_enabled(m_os_log_intervals)) {
    return;
  }
  switch (signpost_type) {
    case SignpostTypes::RUN_MPSGRAPH:
      os_signpost_interval_end(m_os_log_intervals, signpost_id, kSignpostNameRunMPSGraphStr);
      break;
    case SignpostTypes::RUN_KERNEL:
      os_signpost_interval_end(m_os_log_intervals, signpost_id, kSignpostNameRunCustomKernelStr);
      break;
    case SignpostTypes::BLIT_COPY:
      os_signpost_interval_end(m_os_log_intervals, signpost_id, kSignpostNameBlitCopyStr);
      break;
    default:
      AT_ERROR("unknown SignpostType in MPS profiler");
  }
}

MPSProfiler& getMPSProfiler() {
  static std::unique_ptr<MPSProfiler> mps_profiler;
  if (mps_profiler == nullptr) {
    mps_profiler = std::make_unique<MPSProfiler>();
  }
  return *mps_profiler;
}

}  // namespace at::mps
