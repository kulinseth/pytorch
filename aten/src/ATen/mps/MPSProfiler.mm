//  Copyright © 2022 Apple Inc.

#include <c10/util/Exception.h>
#include <ATen/mps/MPSProfiler.h>
#include <malloc/malloc.h>

// these need to be literal strings when passed to os_signpost*()
// function macros; so no LUTs could be used
#define kMPSProfilerSubSystemStr    "PyTorchMPS"
#define kMPSCategoryEventsStr       "Events"
#define kMPSCategoryIntervalsStr    "Intervals"
#define kIntSignpostRunOperationStr "PyTorchOperationIntervals"
#define kIntSignpostBlitCopyStr     "PyTorchCopyIntervals"
#define kIntSignpostCPUFallbacksStr "PyTorchCPUFallbackIntervals"
#define kEvtSignpostRunOperationStr "PyTorchOperationEvents"
#define kEvtSignpostBlitCopyStr     "PyTorchCopyEvents"
#define kEvtSignpostCPUFallbacksStr "PyTorchCPUFallbacksEvents"
#define kEVLogProfileInfoStr        "PYTORCH_MPS_LOG_PROFILE_INFO"
#define kEVTraceSignpostsStr        "PYTORCH_MPS_TRACE_SIGNPOSTS"

namespace at::mps {
namespace Profiler {

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
  currentSigint.sa_handler = nullptr;
  previousSigint.sa_handler = nullptr;

  initialize();
}

MPSProfiler::~MPSProfiler() {
  // first make sure completion handlers are completed
  auto stream = getDefaultMPSStream();
  dispatch_sync(stream->queue(), ^() {
    if (hasPendingCompletionHandlers) {
      stream->synchronize(SyncType::COMMIT_AND_WAIT);
    }
  });
  logProfilingStats();

  if (m_os_log_events) {
    os_release(m_os_log_events);
  }
  if (m_os_log_intervals) {
    os_release(m_os_log_intervals);
  }
}

void MPSProfiler::initialize() {
  if ((m_signpost_types == SignpostTypes::SIGNPOST_NONE) &&
      (m_profile_options & ProfileOptions::INCLUDE_SCHEDULE_INTERVAL)) {
    m_profile_options |= ProfileOptions::ALL_SIGNPOST_INTERVALS;
  }

  if (m_profile_options & (ProfileOptions::ALL_SIGNPOST_EVENTS |
                           ProfileOptions::ALL_SIGNPOST_INTERVALS)) {
    // enable all signposts types
    m_signpost_types |= (SignpostTypes::RUN_OPERATION |
                         SignpostTypes::CPU_FALLBACK |
                         SignpostTypes::BLIT_COPY);

    if (m_profile_options & ProfileOptions::ALL_SIGNPOST_EVENTS) {
      m_profile_options |= ProfileOptions::USE_EVENTS;
    }
    if (m_profile_options & ProfileOptions::ALL_SIGNPOST_INTERVALS) {
      m_profile_options |= ProfileOptions::USE_INTERVALS;
    }
  }

  if (m_log_options & LogOptions::ALL_STATS) {
    m_log_options |= LogOptions::OPERATION_STATS |
                     LogOptions::COPY_STATS |
                     LogOptions::CPU_FALLBACK_STATS;
  }
  // if memory growth tracking is enabled, we need to execute command buffers
  // with waitUntilCompleted to track each operation from completion handlers, separately.
  // Otherwise, the memory usages for "all" the ops encoded in a command buffer
  // would sum up and displayed incorrectly for a single op.
  if (m_log_options & LogOptions::INCLUDE_MEMORY_GROWTH) {
    m_profile_options |= ProfileOptions::WAIT_UNTIL_COMPLETED;
  }

  if (m_signpost_types != SignpostTypes::SIGNPOST_NONE) {
    // if no signpost options passed, use interval mode by default
    if (!(m_profile_options & (ProfileOptions::USE_EVENTS | ProfileOptions::USE_INTERVALS))) {
      m_profile_options |= ProfileOptions::USE_INTERVALS;
    }
    if ((m_profile_options & ProfileOptions::INCLUDE_SCHEDULE_INTERVAL) &&
        (m_profile_options & ProfileOptions::USE_EVENTS)) {
      TORCH_CHECK((m_profile_options & ProfileOptions::USE_INTERVALS),
                  "the option 'INCLUDE_SCHEDULE_INTERVAL' only works for interval-based signposts");
    }

    // technically, it's possible to trace both events and intervals at the same time
    if (m_profile_options & ProfileOptions::USE_EVENTS) {
      if (!m_os_log_events) {
        m_os_log_events = os_log_create(kMPSProfilerSubSystemStr, kMPSCategoryEventsStr);
        TORCH_CHECK(m_os_log_events, "failed to create OS signpost log for events profiler");
      }
      // include GPU and CPU times in metadata for event-based intervals by default, since
      // events are marked in Metal Completion Handlers which outputs GPU time
      m_log_options |= INCLUDE_EXECUTION_TIME;
    }
    if (m_profile_options & ProfileOptions::USE_INTERVALS) {
      if (!m_os_log_intervals) {
        m_os_log_intervals = os_log_create(kMPSProfilerSubSystemStr, kMPSCategoryIntervalsStr);
        TORCH_CHECK(m_os_log_intervals, "failed to create OS signpost log for intervals profiler");
      }
    }
  }

  if (m_log_options & LogOptions::COPY_STATS) {
    if (m_copy_stat_list.empty()) {
      m_copy_stat_list.emplace(CopyInfo::Kind::MPS_TO_MPS, std::make_unique<CopyStat>("MPS to MPS"));
      m_copy_stat_list.emplace(CopyInfo::Kind::MPS_TO_CPU, std::make_unique<CopyStat>("MPS to CPU"));
      m_copy_stat_list.emplace(CopyInfo::Kind::CPU_TO_MPS, std::make_unique<CopyStat>("CPU to MPS"));
    }
  }

  // used to capture sigint signal to log profiling stats
  if (m_log_options & (LogOptions::OPERATION_STATS |
                       LogOptions::COPY_STATS |
                       LogOptions::CPU_FALLBACK_STATS)) {
    if (!currentSigint.sa_handler) {
      currentSigint.sa_handler = &handleIntSignal;
      currentSigint.sa_flags = SA_RESTART;
      sigfillset(&currentSigint.sa_mask);
      if (sigaction(SIGINT, &currentSigint, &previousSigint) == -1) {
        AT_ERROR("Cannot install SIGINT handler for MPSProfiler.");
      }
    }
  }
}

void MPSProfiler::StartTrace(const string& mode, bool waitUntilCompleted) {
  TORCH_CHECK(m_profile_options == ProfileOptions::OPTIONS_NONE,
              "Tracing Signposts is already enabled ");

  std::stringstream ss(mode);
  std::string token;
  while (getline(ss, token, ',')) {
    if (!token.empty()) {
      if (token == "interval") {
        m_profile_options |= ProfileOptions::ALL_SIGNPOST_INTERVALS;
      } else if (token == "event") {
        m_profile_options |= ProfileOptions::ALL_SIGNPOST_EVENTS;
      } else if (token == "log_stats") {
        m_log_api_options = LogOptions::OPERATION_STATS |
                            LogOptions::COPY_STATS |
                            LogOptions::CPU_FALLBACK_STATS;
        // keep m_log_api_options separately to avoid conflicts with env-var log settings
        m_log_options |= m_log_api_options;
      } else {
        AT_ERROR("Invalid Signpost trace mode: ", token);
      }
    }
  }
  if (m_profile_options != ProfileOptions::OPTIONS_NONE) {
    if (waitUntilCompleted) {
      m_profile_options |= ProfileOptions::WAIT_UNTIL_COMPLETED;
    }
  }
  initialize();
}

void MPSProfiler::StopTrace() {
  m_profile_options = ProfileOptions::OPTIONS_NONE;
  m_signpost_types = SignpostTypes::SIGNPOST_NONE;
  // only disable log options that were enabled via the APIs and not env-vars
  m_log_options &= ~m_log_api_options;
}

void MPSProfiler::beginProfileExecution(BaseInfo& info, bool cpuExecution, bool generateSignpost) {
  info.startTimeCPU = getCPUTime();
  if (m_log_options & LogOptions::INCLUDE_MEMORY_GROWTH) {
    info.startCpuMemoryUsage = getMallocedMemorySize();
  }
  const std::string& infoStr = info.toString();
  // see comments in isProfileInfoLoggingEnabled()
  if (isProfileInfoLoggingEnabled(info.type, /*isExecutionEnded*/ false)) {
    fmt::print(stderr, "{}\n", infoStr);
  }
  SignpostTypes signpostType = getSignpostType(info.type);

  if (generateSignpost && (m_signpost_types & signpostType)) {
    if (m_profile_options & ProfileOptions::USE_EVENTS) {
      info.eventSignpostId = generateSignpostId(OS_SIGNPOST_EVENT);
    }
    if (m_profile_options & ProfileOptions::USE_INTERVALS) {
      info.intervalSignpostId = generateSignpostId(OS_SIGNPOST_INTERVAL_BEGIN);
      // if scheduling part is included, we begin the interval early in here,
      // otherwise we begin when the scheduledHandler callback is triggered.
      if ((m_profile_options & ProfileOptions::INCLUDE_SCHEDULE_INTERVAL) || cpuExecution) {
        beginSignpostInterval(signpostType, info.intervalSignpostId, infoStr);
        info.completed = false;
      // for graphs, we add the scheduleHandler in beginProfileGPUInterval()
      } else if (info.type == BaseInfo::Type::KERNEL || info.type == BaseInfo::Type::COPY) {
        addProfilerScheduledHandler(info);
      }
    }
  }
}

void MPSProfiler::endProfileExecution(BaseInfo& info, os_signpost_id_t event_signpost_id,
                                      os_signpost_id_t interval_signpost_id,
                                      double gpuTime, double schedulingTime) {
  const double cpuTime = double(getCPUTime() - info.startTimeCPU) * 1e-6;
  const SignpostTypes signpostType = getSignpostType(info.type);

  if (info.type == BaseInfo::Type::COPY) {
    updateCopyStats(static_cast<CopyInfo&>(info), gpuTime, cpuTime, schedulingTime);
  } else {
    info.totalGpuTime = info.totalGpuTime + gpuTime;
    info.totalCpuTime = info.totalCpuTime + cpuTime;
    info.totalSchedulingTime = info.totalSchedulingTime + schedulingTime;
  }
  // if Kernel time is included, we add it to gpuTime in metadata
  if (gpuTime > 0.0 && (m_log_options & LogOptions::INCLUDE_KERNEL_TIME)) {
    gpuTime += schedulingTime;
  }
  size_t memGrowth = 0;
  if (m_log_options & LogOptions::INCLUDE_MEMORY_GROWTH) {
    size_t current_mem_usage = getMallocedMemorySize();
    if (current_mem_usage > info.startCpuMemoryUsage) {
      memGrowth = current_mem_usage - info.startCpuMemoryUsage;
    }
  }
  const std::string& infoStr = info.toString(gpuTime, cpuTime, memGrowth);
  // see comments in isProfileInfoLoggingEnabled()
  if (isProfileInfoLoggingEnabled(info.type, /*isExecutionEnded*/ true)) {
    fmt::print(stderr, "{}\n", infoStr);
  }
  // it is possible to use both interval and event based signposts at the same time
  if ((m_profile_options & ProfileOptions::USE_EVENTS) && event_signpost_id) {
    emitSignpostEvent(signpostType, event_signpost_id, infoStr);
  }
  // GPU time for signpost intervals is calculated based on its duration
  if ((m_profile_options & ProfileOptions::USE_INTERVALS) && interval_signpost_id) {
    endSignpostInterval(signpostType, interval_signpost_id);
  }
  info.completed = true;
}

uint64_t MPSProfiler::beginProfileKernel(const void* handle, const std::string& strKey, bool isGraph) {
  // only do profiling if operation execution profiling or logging are enabled
  if (!isOperationProfilingEnabled()) {
    return 0;
  }
  if (m_op_info_list.count(uintptr_t(handle)) == 0) {
    auto opInfo = std::make_unique<OperationInfo>(handle, isGraph,
                         isGraph ? ++m_graph_counter : ++m_kernel_counter, strKey);
    m_op_info_list.emplace(opInfo->handle, std::move(opInfo));
  }
  auto& opInfo = *m_op_info_list[uintptr_t(handle)];
  opInfo.strKey.assign(strKey);
  opInfo.runCount++;
  beginProfileExecution(opInfo);

  return opInfo.profileId;
}

uint64_t MPSProfiler::beginProfileKernel(const void* handle, const std::string& kernelName, const TensorList& tensors) {
  if (isOperationProfilingEnabled()) {
    const bool includeBufferId = m_log_options & LogOptions::INCLUDE_BUFFER_ID;
    std::string profilerStrKey = OperationInfo::buildKernelString(kernelName, tensors, includeBufferId);
    return beginProfileKernel(handle, profilerStrKey, false);
  }
  return 0;
}

void MPSProfiler::beginProfileGPUInterval(const void* handle) {
  // this function is only relevant for interval-based Signposts which exclude
  // schedule time (only includes GPU run time)
  if (!(m_profile_options & ProfileOptions::USE_INTERVALS) ||
       (m_profile_options & ProfileOptions::INCLUDE_SCHEDULE_INTERVAL)) {
    return;
  }
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(m_op_info_list.count(uintptr_t(handle)), "Failed to get operation information!");
  auto& opInfo = *m_op_info_list[uintptr_t(handle)];
  // this begins the interval when scheduling the execution is
  // completed already (i.e., scheduling excluded from interval)
  addProfilerScheduledHandler(opInfo);
}

void MPSProfiler::endProfileKernel(const void* handle, SyncType syncType) {
  // only do profiling if operation execution profiling or logging are enabled
  if (!isOperationProfilingEnabled()) {
    return;
  }
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(m_op_info_list.count(uintptr_t(handle)), "Failed to get operation information!");
  auto& opInfo = *m_op_info_list[uintptr_t(handle)];
  addProfilerCompletedHandler(opInfo, syncType);
}

uint64_t MPSProfiler::beginProfileCPUFallback(const std::string& opName, const TensorList& tensors) {
  if (m_cpu_fb_info_list.count(opName) == 0) {
    auto cpuFbInfo = std::make_unique<CpuFbInfo>(++m_cpu_fb_counter, opName);
    m_cpu_fb_info_list.emplace(opName, std::move(cpuFbInfo));
  }
  auto& cpuFbInfo = *m_cpu_fb_info_list[opName];
  cpuFbInfo.runCount++;
  const bool includeBufferId = m_log_options & LogOptions::INCLUDE_BUFFER_ID;
  cpuFbInfo.strKey = OperationInfo::buildKernelString(opName, tensors, includeBufferId);
  cpuFbInfo.updateCopyOverhead(tensors);
  beginProfileExecution(cpuFbInfo, true);

  return cpuFbInfo.profileId;
}

void MPSProfiler::endProfileCPUFallback(const std::string& opName) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(m_cpu_fb_info_list.count(opName), "Failed to get CPU Fallback information!");
  auto& cpuFbInfo = *m_cpu_fb_info_list[opName];
  endProfileExecution(cpuFbInfo, cpuFbInfo.eventSignpostId,
                      cpuFbInfo.intervalSignpostId, 0, 0);
}

uint64_t MPSProfiler::beginProfileCopy(const void* srcBuffer, const void* dstBuffer,
                                       const OptionalTensorRef srcTensor,
                                       const OptionalTensorRef dstTensor,
                                       size_t length, bool isNonBlocking, bool usesBlitter) {
  if (!isCopyProfilingEnabled()) {
    return 0;
  }
  const bool includeBufferId = m_log_options & LogOptions::INCLUDE_BUFFER_ID;
  const uint64_t profileId = ++m_copy_counter;
  auto copyInfo = std::make_unique<CopyInfo>(dstBuffer, length, profileId, isNonBlocking, usesBlitter);
  copyInfo->srcStrKey = CopyInfo::buildTensorString(srcBuffer, srcTensor, includeBufferId);
  copyInfo->dstStrKey = CopyInfo::buildTensorString(dstBuffer, dstTensor, includeBufferId);
  copyInfo->kind = CopyInfo::getCopyKind(srcBuffer, dstBuffer, srcTensor, dstTensor);
  // don't generate signposts if the non-blocking copy is not using the blitter
  const bool generateSignpost = usesBlitter || !isNonBlocking;
  beginProfileExecution(*copyInfo, !usesBlitter, generateSignpost);
  // this should not happen since we erase the copy info after profiling/logging it.
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(m_copy_info_list.count(profileId) == 0);
  // the copy info isn't retained in the list, so we erase the completed ones
  for (auto it = m_copy_info_list.begin(), last = m_copy_info_list.end(); it != last;) {
    if (it->second->completed) {
      it = m_copy_info_list.erase(it);
    } else {
      ++it;
    }
  }
  m_copy_info_list.emplace(profileId, std::move(copyInfo));

  return profileId;
}

void MPSProfiler::endProfileCopy(uint64_t profileId, SyncType syncType) {
  // this is just an identifier, and not used to access memory
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(m_copy_info_list.count(profileId), "Failed to get copy information!");
  auto& copyInfo = *m_copy_info_list[profileId];
  if (copyInfo.usesBlitter) {
    addProfilerCompletedHandler(copyInfo, syncType);
  } else {
    endProfileExecution(copyInfo, copyInfo.eventSignpostId, copyInfo.intervalSignpostId, 0, 0);
  }
}

void MPSProfiler::addProfilerScheduledHandler(BaseInfo& info) {
  const SignpostTypes signpostType = getSignpostType(info.type);
  const os_signpost_id_t intervalSignpostId = info.intervalSignpostId;

  auto m_stream = getDefaultMPSStream();
  // NOTE: the following block isn't thread-safe
  [m_stream->commandBuffer() addScheduledHandler:^(id<MTLCommandBuffer> cb) {
    // begin the interval once scheduling has completed (if INCLUDE_SCHEDULE_INTERVAL flag is disabled)
    beginSignpostInterval(signpostType, intervalSignpostId, info.toString());
    info.completed = false;
  }];
}

void MPSProfiler::updateCopyStats(const CopyInfo& copyInfo, double gpuTime, double cpuTime, double schedulingTime) {
  if (!(m_log_options & LogOptions::COPY_STATS)) {
    return;
  }
  auto& copyStat = *m_copy_stat_list[copyInfo.kind];
  copyStat.totalCount++;
  copyStat.length += copyInfo.length;
  copyStat.totalGpuTime = copyStat.totalGpuTime + gpuTime;
  copyStat.totalCpuTime = copyStat.totalCpuTime + cpuTime;
  copyStat.totalSchedulingTime = copyStat.totalSchedulingTime + schedulingTime;
  if (copyInfo.length <= sizeof(int64_t)) {
    copyStat.scalarsCount++;
    copyStat.scalarsGpuTime = copyStat.scalarsGpuTime + gpuTime;
  }
  copyStat.blockingCount += !copyInfo.isNonBlocking ? 1 : 0;
  copyStat.memcpyCount += !copyInfo.usesBlitter ? 1 : 0;
}

void MPSProfiler::addProfilerCompletedHandler(BaseInfo& info, SyncType syncType) {
  const os_signpost_id_t intervalSignpostId = info.intervalSignpostId;
  const os_signpost_id_t eventSignpostId = info.eventSignpostId;

  // signpost ID is used only for interval-based signposts, and must be non-zero
  if (m_profile_options & ProfileOptions::USE_INTERVALS) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(intervalSignpostId, "Signpost interval has no identifier!");
  }
  // reset signpostIds for sanity check on next call
  info.intervalSignpostId = 0;
  info.eventSignpostId = 0;
  hasPendingCompletionHandlers = true;

  auto m_stream = getDefaultMPSStream();
  // NOTE: the following block isn't thread-safe
  [m_stream->commandBuffer() addCompletedHandler:^(id<MTLCommandBuffer> cb) {
    CFTimeInterval gpuTime = (cb.GPUEndTime - cb.GPUStartTime) * 1000.0;
    CFTimeInterval schedulingTime = (cb.kernelEndTime - cb.kernelStartTime) * 1000.0;

    endProfileExecution(info, eventSignpostId, intervalSignpostId, gpuTime, schedulingTime);
    hasPendingCompletionHandlers = false;
  }];

  m_stream->synchronize((m_profile_options & ProfileOptions::WAIT_UNTIL_COMPLETED) ?
                        SyncType::COMMIT_AND_WAIT : syncType);
}

void MPSProfiler::logOperationsProfilingStats(std::FILE* f) const {
  if (m_op_info_list.empty()) {
    // this is not an error, but to let the user know that the
    // LogOptions::KERNEL_STATS that they passed to EV is not yielding anything.
    fmt::print(f, "There are no MPS operations logged for profiling\n");
    return;
  }
  // dump the ops info into a vector to sort them
  std::vector<OperationInfo*> opsList;
  std::transform(m_op_info_list.begin(), m_op_info_list.end(),
                 std::back_inserter(opsList),
                 [](auto& opInfo){ return opInfo.second.get();} );

  // sort based on "Mean GPU time" in descending order
  std::sort(opsList.begin(), opsList.end(),
            [](const OperationInfo* a, const OperationInfo* b) {
              return (a->totalGpuTime / double(a->runCount)) > (b->totalGpuTime / double(b->runCount));
            });
  // print the table of operation profiling stats
  fmt::print(f, "\n{:-^200}\n{:^6}|{:^7}|{:^10}|{:^11}|{:^10}|{:^11}| {}\n{:-^200}\n",
             fmt::format(" MPS Operations Profiling: {} graphs, {} kernels (times in milliseconds) ",
                         m_graph_counter, m_kernel_counter),
             "ID", "#Runs", "Mean CPU", "Mean KRNL", "Mean GPU", "Total GPU", "Operation Name", "");

  for (const auto& opInfo : opsList) {
    fmt::print(f, "{:^7}{:^8}{:^11}{:^12}{:^11}{:^12} {}\n",
               fmt::format("{}{}", opInfo->type == BaseInfo::Type::GRAPH ? "G": "K", opInfo->profileId),
               opInfo->runCount,
               fmt::format("{:.3f}", opInfo->totalCpuTime / double(opInfo->runCount)),
               fmt::format("{:.3f}", opInfo->totalSchedulingTime / double(opInfo->runCount)),
               fmt::format("{:.3f}", opInfo->totalGpuTime / double(opInfo->runCount)),
               fmt::format("{:.2f}", opInfo->totalGpuTime),
               opInfo->strKey);
  }
}

void MPSProfiler::logCPUFallbackProfilingStats(std::FILE* f) const {
  if (m_cpu_fb_info_list.empty()) {
    // this is not an error, but to let the user know that the
    // LogOptions::KERNEL_STATS that they passed to EV is not yielding anything.
    fmt::print(f, "There are no CPU Fallbacks logged for profiling\n");
    return;
  }
  size_t totalCopyOverhead = 0;
  size_t totalRunCount = 0;
  double totalCPUTime = 0.;
  // dump the map's info into a vector to sort them
  std::vector<CpuFbInfo*> cpuFbList;
  std::transform(m_cpu_fb_info_list.begin(), m_cpu_fb_info_list.end(),
                 std::back_inserter(cpuFbList),
                 [&](auto& cpuFbInfo) {
                  auto cpuFbInfoPtr = cpuFbInfo.second.get();
                  totalRunCount += cpuFbInfoPtr->runCount;
                  totalCopyOverhead += cpuFbInfoPtr->totalCopyOverhead;
                  totalCPUTime += cpuFbInfoPtr->totalCpuTime;
                  return cpuFbInfoPtr;
                 });

  // sort based on "Mean CPU time" in descending order
  std::sort(cpuFbList.begin(), cpuFbList.end(),
            [](const CpuFbInfo* a, const CpuFbInfo* b) {
              return (a->totalCpuTime / double(a->runCount)) > (b->totalCpuTime / double(b->runCount));
            });

  // print the table of CPU Fallback profiling stats
  fmt::print(f, "\n{:-^150}\n{:^5}|{:^7}|{:^14}|{:^15}|{:^15}| {}\n{:-^150}\n",
             fmt::format(" CPU Fallback Profiling: Total {} Runs, {:.2f} ms, {} Copies ",
                         totalRunCount, totalCPUTime,
                         BaseInfo::formatSize(totalCopyOverhead)),
             "ID", "#Runs", "Mean CPU(ms)", "Total CPU(ms)", "Copy Overhead", "Operation Name", "");

  for (const auto& cpuFbInfo : cpuFbList) {
    fmt::print(f, "{:^6}{:^8}{:^15}{:^16}{:^16} {}\n",
               cpuFbInfo->profileId, cpuFbInfo->runCount,
               fmt::format("{:.3f}", cpuFbInfo->totalCpuTime / double(cpuFbInfo->runCount)),
               fmt::format("{:.2f}", cpuFbInfo->totalCpuTime),
               BaseInfo::formatSize(cpuFbInfo->totalCopyOverhead),
               cpuFbInfo->opName);
  }
}

void MPSProfiler::logCopyProfilingStats(std::FILE* f) const {
  size_t totalCopiesCount = 0;
  size_t totalCopySize = 0;
  size_t totalScalarCopyCount = 0;

  for (const auto& copyStatPair : m_copy_stat_list) {
    const auto& copyStat = *copyStatPair.second;
    totalCopiesCount += copyStat.totalCount;
    totalCopySize += copyStat.length;
    totalScalarCopyCount += copyStat.scalarsCount;
  }
  if (totalCopiesCount == 0) {
    // this is not an error, but to let the user know that the
    // LogOptions::COPY_STATS that they passed to EV is not yielding anything.
    fmt::print(f, "There are no copies logged for profiling\n");
    return;
  }

  // print the table of copy profiling stats
  fmt::print(f, "\n{:-^160}\n{:^12}|{:^10}|{:^17}|{:^15}|{:^16}|{:^15}|{:^9}|{:^13}|{:^10}|{:^8}\n{:-^160}\n",
             fmt::format(" MPS Copy Profiling: {} total copies ({}), {} scalar copies ",
                         totalCopiesCount, BaseInfo::formatSize(totalCopySize), totalScalarCopyCount),
             "Kind", "Total#", "Total Size", "Total CPU(ms)", "Total KRNL(ms)", "Total GPU(ms)", "Scalars",
             "Scalars GPU", "Blocking", "memcpy", "");

  for (const auto& copyStatPair : m_copy_stat_list) {
    const auto& copyStat = *copyStatPair.second;
    if (copyStat.totalCount > 0) {
      fmt::print(f, "{:^13}{:^11}{:^18}{:^16}{:^17}{:^16}{:^10}{:^14}{:^11}{:^9}\n",
                copyStat.kindStr, copyStat.totalCount,
                BaseInfo::formatSize(copyStat.length),
                fmt::format("{:.2f}", copyStat.totalCpuTime),
                fmt::format("{:.2f}", copyStat.totalSchedulingTime),
                fmt::format("{:.2f}", copyStat.totalGpuTime), copyStat.scalarsCount,
                fmt::format("{:.2f} %", copyStat.totalGpuTime > 0.0 ?
                (1.0 - ((copyStat.totalGpuTime - copyStat.scalarsGpuTime) /
                copyStat.totalGpuTime)) * 100.0 : 0.0),
                copyStat.blockingCount, copyStat.memcpyCount);
    }
  }
}

void MPSProfiler::logProfilingStats() {
  if (hasLoggedStats.exchange(true)) {
    return;
  }
  // either the env-var or startTrace() API might have logged the stats
  uint32_t log_options = m_log_options | m_log_api_options;
  // logs kernel profiling stats when the process ends (if enabled).
  if (log_options & LogOptions::OPERATION_STATS) {
    logOperationsProfilingStats(stderr);
  }
  // logs CPU Fallback profiling stats when the process ends (if enabled).
  if (log_options & LogOptions::CPU_FALLBACK_STATS) {
    logCPUFallbackProfilingStats(stderr);
  }
  // logs copies profiling stats when the process ends (if enabled).
  if (log_options & LogOptions::COPY_STATS) {
    logCopyProfilingStats(stderr);
  }
}

bool MPSProfiler::isProfileInfoLoggingEnabled(BaseInfo::Type infoType, bool isExecutionEnded) {
  bool isInfoLoggingEnabled = false;
  // logging the operations, copies, cpu fallbacks info during the execution
  // is enabled via the env-var defined in kEVLogProfileInfoStr
  switch (infoType) {
    case BaseInfo::Type::GRAPH:
    case BaseInfo::Type::KERNEL:
      isInfoLoggingEnabled = (m_log_options & LogOptions::OPERATION_INFO);
      break;
    case BaseInfo::Type::COPY:
      isInfoLoggingEnabled = (m_log_options & LogOptions::COPY_INFO);
      break;
    case BaseInfo::Type::CPU_FALLBACK:
      isInfoLoggingEnabled = (m_log_options & LogOptions::CPU_FALLBACK_INFO);
      break;
    default:
      AT_ERROR("invalid profiling info type");
  }
  if (!isInfoLoggingEnabled) {
    return false;
  }
  // if GPU/Kernel times are included then log info when op execution ends
  bool logWhenExecutionEnds = m_log_options & (LogOptions::INCLUDE_EXECUTION_TIME |
                                               LogOptions::INCLUDE_KERNEL_TIME |
                                               LogOptions::INCLUDE_MEMORY_GROWTH);
  return isExecutionEnded ? logWhenExecutionEnds : !logWhenExecutionEnds;
}

void MPSProfiler::emitSignpostEvent(SignpostTypes signpost_type, os_signpost_id_t signpost_id,
                                    const std::string& msg_str) const {
  if (!(m_signpost_types & signpost_type) || !signpost_id ||
      !m_os_log_events || !os_signpost_enabled(m_os_log_events)) {
    return;
  }
  const char *msg = msg_str.c_str();

  // need to use switch-case as the signpost names must be literal strings
  switch (signpost_type) {
    case SignpostTypes::RUN_OPERATION:
      os_signpost_event_emit(m_os_log_events, signpost_id, kEvtSignpostRunOperationStr, "%s", msg);
      break;
    case SignpostTypes::BLIT_COPY:
      os_signpost_event_emit(m_os_log_events, signpost_id, kEvtSignpostBlitCopyStr, "%s", msg);
      break;
    case SignpostTypes::CPU_FALLBACK:
      os_signpost_event_emit(m_os_log_events, signpost_id, kEvtSignpostCPUFallbacksStr, "%s", msg);
      break;
    default:
      AT_ERROR("unknown SignpostType in MPS profiler");
  }
}

void MPSProfiler::beginSignpostInterval(SignpostTypes signpost_type, os_signpost_id_t signpost_id,
                                        const std::string& msg_str) const {
  if (!(m_signpost_types & signpost_type) || !signpost_id ||
      !m_os_log_intervals || !os_signpost_enabled(m_os_log_intervals)) {
    return;
  }
  const char *msg = msg_str.c_str();

  switch (signpost_type) {
    case SignpostTypes::RUN_OPERATION:
      os_signpost_interval_begin(m_os_log_intervals, signpost_id, kIntSignpostRunOperationStr, "%s", msg);
      break;
    case SignpostTypes::BLIT_COPY:
      os_signpost_interval_begin(m_os_log_intervals, signpost_id, kIntSignpostBlitCopyStr, "%s", msg);
      break;
    case SignpostTypes::CPU_FALLBACK:
      os_signpost_interval_begin(m_os_log_intervals, signpost_id, kIntSignpostCPUFallbacksStr, "%s", msg);
      break;
    default:
      AT_ERROR("unknown SignpostType in MPS profiler");
  }
}

void MPSProfiler::endSignpostInterval(SignpostTypes signpost_type, os_signpost_id_t signpost_id) const {
  if (!m_os_log_intervals || !os_signpost_enabled(m_os_log_intervals)) {
    return;
  }
  switch (signpost_type) {
    case SignpostTypes::RUN_OPERATION:
      os_signpost_interval_end(m_os_log_intervals, signpost_id, kIntSignpostRunOperationStr);
      break;
    case SignpostTypes::BLIT_COPY:
      os_signpost_interval_end(m_os_log_intervals, signpost_id, kIntSignpostBlitCopyStr);
      break;
    case SignpostTypes::CPU_FALLBACK:
      os_signpost_interval_end(m_os_log_intervals, signpost_id, kIntSignpostCPUFallbacksStr);
      break;
    default:
      AT_ERROR("unknown SignpostType in MPS profiler");
  }
}

os_signpost_id_t MPSProfiler::generateSignpostId(os_signpost_type_t signpostType, const void* ptr) {
  os_log_t os_log = signpostType == OS_SIGNPOST_EVENT ? m_os_log_events : m_os_log_intervals;
  if (ptr) {
    return os_signpost_id_make_with_pointer(os_log, ptr);
  }
  return os_signpost_id_generate(os_log);
}

MPSProfiler::SignpostTypes MPSProfiler::getSignpostType(BaseInfo::Type infoType) {
  switch (infoType) {
    case BaseInfo::Type::GRAPH:
    case BaseInfo::Type::KERNEL:
      return SignpostTypes::RUN_OPERATION;
    case BaseInfo::Type::COPY:
      return SignpostTypes::BLIT_COPY;
    case BaseInfo::Type::CPU_FALLBACK:
      return SignpostTypes::CPU_FALLBACK;
    default:
      AT_ERROR("invalid profiling info type");
  }
}

void MPSProfiler::handleIntSignal(int signal) {
  getMPSProfiler().logProfilingStats();
  if (previousSigint.sa_handler) {
    previousSigint.sa_handler(signal);
  }
}

size_t MPSProfiler::getMallocedMemorySize() {
  malloc_statistics_t stats;
  bzero(&stats,sizeof(stats));
  malloc_zone_statistics(nullptr, &stats);
  return stats.size_in_use;
}

// used to capture sigint signal to log profiling stats
struct sigaction MPSProfiler::currentSigint{};
struct sigaction MPSProfiler::previousSigint{};

} // namespace Profiler

Profiler::MPSProfiler& getMPSProfiler() {
  static auto mps_profiler = std::make_unique<Profiler::MPSProfiler>();
  return *mps_profiler;
}

}  // namespace at::mps
