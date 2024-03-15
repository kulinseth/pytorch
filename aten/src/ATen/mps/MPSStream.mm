//  Copyright © 2022 Apple Inc.

#include <ATen/mps/MPSStream.h>
#include <ATen/mps/MPSProfiler.h>
#include <ATen/mps/MPSAllocatorInterface.h>

@interface MPSGraphExecutionDescriptor ()
@property (readwrite, atomic) BOOL enableCommitAndContinue;
@end

namespace at {
namespace mps {

// threshold to perform adaptive commit if the accumulated size
// of resources encoded on the command buffer exceeds that.
static const size_t kCmdBufAdaptiveCommitThreshold = MB(64);

//-----------------------------------------------------------------
//  MPSStream
//-----------------------------------------------------------------

MPSStream::MPSStream(Stream stream) : _stream(stream) {
  _commandQueue = [MPSDevice::getInstance()->device() newCommandQueue];
  TORCH_CHECK(_stream.device_type() == DeviceType::MPS);
  _serialQueue = dispatch_queue_create("metal gpu stream", nullptr);
  _executionDescriptor = [MPSGraphExecutionDescriptor new];
  _executableExecutionDescriptor = [MPSGraphExecutableExecutionDescriptor new];
  // disable commitAndContinue if Signpost tracing is enabled
  if (getMPSProfiler().isSignpostTracingEnabled()) {
    _enableCommitAndContinue = false;
  }
  // internal CommitAndContinue heuristic of MPSGraph is disabled, and we
  // control it via Adaptive Commit in PyTorch-side
  _executionDescriptor.enableCommitAndContinue = false;
}

MPSStream::~MPSStream() {
  [_commandQueue release];
  _commandQueue = nil;
  [_executionDescriptor release];
  [_executableExecutionDescriptor release];

  _executionDescriptor = nil;
  _executableExecutionDescriptor = nil;

  assert(_commandBuffer == nil);
}

MPSCommandBuffer* MPSStream::commandBuffer() {
  if (!_commandBuffer) {
    _commandBuffer = [MPSCommandBuffer commandBufferFromCommandQueue:_commandQueue].retain;
  }

  return _commandBuffer;
}

id<MTLComputeCommandEncoder> MPSStream::commandEncoder() {
  if (!_commandEncoder) {
    _commandEncoder = [commandBuffer() computeCommandEncoder].retain;
  }

  return _commandEncoder;
}

void MPSStream::synchronize(SyncType syncType) {
  endKernelCoalescing();
  switch(syncType) {
    case SyncType::NONE:
      // typically in GPU to GPU copies we won't commit explicitly
      break;
    case SyncType::COMMIT:
      commit();
      break;
    case SyncType::COMMIT_ADAPTIVE:
      // the adaptive commit only commits if we hit the low watermark memory threshold,
      // or when the sizes attached to the active command buffer exceeds the threshold
      if (getIMPSAllocator()->getLowWatermarkValue() <= 0 ||
          _commandBufferResourceSize > kCmdBufAdaptiveCommitThreshold) {
        [commandBuffer() addCompletedHandler:(^(id <MTLCommandBuffer>) {
          getIMPSAllocator()->freeInactiveBuffers();
        })];
        commit();
      }
      break;
    case SyncType::COMMIT_AND_WAIT:
      commitAndWait();
      break;
    case SyncType::COMMIT_AND_CONTINUE:
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(_enableCommitAndContinue,
                                       "CommitAndContinue is called but it is disabled globally!");
      commitAndContinue();
      break;
  }
}

void MPSStream::commit() {
  if (_enableCommitAndContinue) {
    [commandBuffer() commitAndContinue];
  } else {
    flush();
  }
  // reset the accumulated resource sizes for command buffer
  _commandBufferResourceSize = 0;
}

void MPSStream::commitAndWait() {
  if (_prevCommandBuffer) {
    // the previous command buffer (if exists) has already been committed,
    // so we just wait until it's completed and then dispose it.
    [_prevCommandBuffer waitUntilCompleted];
    [_prevCommandBuffer release];
    _prevCommandBuffer = nil;
  }

  if (_commandBuffer) {
    if (_enableCommitAndContinue) {
      // no need to release the command buffer with CommitAndContinue
      // This improves the performance by eliminating the overhead of recreating
      // command buffers, and avoiding distruption to commitAndContinue's internal cache
      id<MTLCommandBuffer> rootCommandBuffer = _commandBuffer.rootCommandBuffer;
      [_commandBuffer commitAndContinue];
      [rootCommandBuffer waitUntilCompleted];
    } else {
      [_commandBuffer commit];
      [_commandBuffer waitUntilCompleted];
      [_commandBuffer release];
      _commandBuffer = nil;
    }
    // reset the accumulated resource sizes for command buffer
    _commandBufferResourceSize = 0;
    // after waiting, it's a good time to free some pending inactive buffers
    getIMPSAllocator()->freeInactiveBuffers();
  }
}

void MPSStream::commitAndContinue() {
  assert(_commandBuffer);
  [_commandBuffer commitAndContinue];
}

void MPSStream::commitAdaptive(const TensorList& inputTensors,
                               const TensorList& outputTensors, void* profilerHandle) {
  std::vector<void*> buffers;
  bool isOutputTensor = false;

  for (const auto& tensorsList : {inputTensors, outputTensors}) {
    for (const auto& tensor : tensorsList) {
      _commandBufferResourceSize += tensor.nbytes();
      if (isOutputTensor) {
        buffers.push_back(tensor.storage().data());
      }
    }
    isOutputTensor = true;
  }

  SyncType syncType = SyncType::COMMIT_ADAPTIVE;
  if (!buffers.empty()) {
    bool recordedEvents = getIMPSAllocator()->recordEvents(buffers);
    if (recordedEvents) {
      syncType = SyncType::COMMIT;
    }
  }
  auto& profiler = getMPSProfiler();
  if (profiler.isOperationProfilingEnabled() && profilerHandle) {
    profiler.endProfileKernel(profilerHandle, syncType);
  } else {
    synchronize(syncType);
  }
}

void MPSStream::endKernelCoalescing() {
  if (_commandEncoder) {
    [_commandEncoder endEncoding];
    [_commandEncoder release];
    _commandEncoder = nil;
  }
}

void MPSStream::flush() {
  if (_commandBuffer) {
    [_commandBuffer commit];
    // if commitAndContinue is disabled (e.g., for Profiler), we keep the command
    // buffer so we could wait on it later, if required.
    if (!_enableCommitAndContinue) {
      _prevCommandBuffer = _commandBuffer;
    } else {
      [_commandBuffer release];
    }
    _commandBuffer = nil;
  }
}

void MPSStream::addCompletedHandler(MTLCommandBufferHandler block, SyncType syncType) {
 dispatch_sync(_serialQueue, ^() {
    @autoreleasepool {
      [commandBuffer() addCompletedHandler:block];
      synchronize(syncType);
    }
  });
}

void MPSStream::fill(id<MTLBuffer> buffer, uint8_t value, size_t length, size_t offset, SyncType syncType)
{
  TORCH_INTERNAL_ASSERT(length >= offset);
  if (length == 0) return;
  dispatch_sync(_serialQueue, ^() {
    @autoreleasepool {
      endKernelCoalescing();
      id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer() blitCommandEncoder];

      [blitEncoder fillBuffer:buffer
                        range:NSMakeRange(offset, length)
                        value:value];
      [blitEncoder endEncoding];
      synchronize(syncType);
    }
  });
}

void MPSStream::copy(id<MTLBuffer> srcBuffer, id<MTLBuffer> dstBuffer,
                    size_t length, size_t srcOffset, size_t dstOffset,
                    uint64_t profileId, SyncType syncType) {
  dispatch_sync(_serialQueue, ^() {
    @autoreleasepool {
      endKernelCoalescing();
      id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer() blitCommandEncoder];

      [blitEncoder copyFromBuffer:srcBuffer
                     sourceOffset:(NSUInteger)srcOffset
                         toBuffer:dstBuffer
                destinationOffset:(NSUInteger)dstOffset
                             size:(NSUInteger)length];
      [blitEncoder endEncoding];

      // profilerId has a value only if copy profiling is enabled
      if (profileId) {
        getMPSProfiler().endProfileCopy(profileId, syncType);
      } else {
        synchronize(syncType);
      }
    }
  });
}

void MPSStream::copy_and_sync(id<MTLBuffer> srcBuffer, id<MTLBuffer> dstBuffer, size_t length,
                              size_t srcOffset, size_t dstOffset, bool non_blocking, uint64_t profileId) {
  copy(srcBuffer, dstBuffer, length, srcOffset, dstOffset, profileId,
       !non_blocking ? SyncType::COMMIT_AND_WAIT : SyncType::COMMIT);
}

void MPSStream::executeMPSGraph(MPSGraph* mpsGraph, NSDictionary* feeds, NSDictionary* results,
                                SyncType syncType, MPSGraphExecutable* executable) {
  auto& profiler = getMPSProfiler();
  const bool isGraphProfilingEnabled = profiler.isOperationProfilingEnabled();

  dispatch_sync(_serialQueue, ^() {
    @autoreleasepool {
      endKernelCoalescing();
      if (isGraphProfilingEnabled) {
        // this function call is only relevant for interval-based Signposts
        // which exclude schedule time (only includes GPU run time)
        profiler.beginProfileGPUInterval(mpsGraph);
      }

      if (executable) {
        NSMutableArray *inputsArray  = [NSMutableArray arrayWithCapacity:[feeds count]];
        NSMutableArray *resultsArray = [NSMutableArray arrayWithCapacity:[results count]];
        NSUInteger inputIndex = 0, ouputIndex = 0;
        for (MPSGraphTensor *tensor in [executable feedTensors]) {
          inputsArray[inputIndex++] = feeds[tensor];
        }

        for (MPSGraphTensor *tensor in [executable targetTensors]) {
          resultsArray[ouputIndex++] = results[tensor];
        }

        [executable encodeToCommandBuffer:commandBuffer()
                              inputsArray:inputsArray
                             resultsArray:resultsArray
                      executionDescriptor:_executableExecutionDescriptor];
      } else {
        // note: CommitAndContinue feature is enabled/disabled via "_executionDescriptor"
        [mpsGraph encodeToCommandBuffer:commandBuffer()
                                  feeds:feeds
                       targetOperations:nil
                      resultsDictionary:results
                    executionDescriptor:_executionDescriptor];
      }
      // if commitAndContinue is disabled, we need to always commit manually after encoding
      SyncType _syncType = _enableCommitAndContinue == false ? SyncType::COMMIT : syncType;

      bool recordedEvents = updateCommandBufferResourceSize([feeds allValues], [results allValues]);
      if (recordedEvents && _syncType != SyncType::COMMIT_AND_WAIT) {
        _syncType = SyncType::COMMIT;
      }
      // check if graph execution profiling is enabled
      if (isGraphProfilingEnabled) {
        // with profiler enabled, we commit after adding the completedHandler in MPSProfiler
        profiler.endProfileKernel(mpsGraph, _syncType);
      } else {
        synchronize(_syncType);
      }
    }
 });
}

bool MPSStream::updateCommandBufferResourceSize(NSArray<MPSGraphTensorData*> *feeds,
                                                NSArray<MPSGraphTensorData*> *results) {
  std::vector<void*> buffers;
  for (const auto& tensorsDataList : {feeds, results}) {
    for (MPSGraphTensorData* tensorData in tensorsDataList) {
      _commandBufferResourceSize += tensorData.mpsndarray.resourceSize;
      if (tensorsDataList == results && _activeResources.count(tensorData)) {
        buffers.push_back(_activeResources[tensorData]);
      }
    }
  }
  bool recordedEvents = false;
  if (!buffers.empty()) {
    recordedEvents = getIMPSAllocator()->recordEvents(buffers);
  }
  return recordedEvents;
}

void MPSStream::addActiveResource(MPSGraphTensorData* tensorData, void* buffer) {
  _activeResources[tensorData] = buffer;
}

//-----------------------------------------------------------------
//  MPSStreamImpl
//-----------------------------------------------------------------

MPSStream* MPSStreamImpl::_stream = nullptr;

MPSStream* MPSStreamImpl::getInstance() {
  if (_stream == nullptr) {
    _stream =
        new MPSStream(Stream(Stream::UNSAFE, c10::Device(DeviceType::MPS), 0));
  }
  return _stream;
}

MPSStreamImpl::MPSStreamImpl() {}

MPSStream* getCurrentMPSStream() {
  return getDefaultMPSStream();
}

MPSStream* getDefaultMPSStream() {
  return MPSStreamImpl::getInstance();
}

//-----------------------------------------------------------------
//  MPSEvent
//-----------------------------------------------------------------

MPSEvent::MPSEvent(bool needsListener) :
    _stream(getDefaultMPSStream()),
    _event([_stream->device() newSharedEvent]) {

  if (needsListener) {
    _listener = [[MTLSharedEventListener alloc] init];
  }
}

MPSEvent::~MPSEvent() {
  if (_event) {
    [_event release];
    _event = nil;
  }
  if (_listener) {
    [_listener release];
    _listener = nil;
  }
}

void MPSEvent::recordEventLocked(bool syncEvent) {
  // active encoders must end before encoding or waiting
  _stream->endKernelCoalescing();
  ++_signalCounter;

  id<MTLCommandBuffer> commandBuffer = _stream->commandBuffer();
  [commandBuffer encodeSignalEvent:_event value:_signalCounter];
  if (syncEvent) {
    _stream->synchronize(SyncType::COMMIT);
  }
}

bool MPSEvent::waitForEventLocked(bool syncEvent) {
  // check if event is not recorded yet
  if (_event.signaledValue >= _signalCounter) {
    return false;
  }
  // active encoders must end before encoding or waiting
  _stream->endKernelCoalescing();
  id<MTLCommandBuffer> commandBuffer = _stream->commandBuffer();
  [commandBuffer encodeWaitForEvent:_event value:_signalCounter];
  if (syncEvent) {
    _stream->synchronize(SyncType::COMMIT);
  }
  return true;
}

bool MPSEvent::notifyEventLocked(bool syncEvent, MTLSharedEventNotificationBlock block) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(_listener);
  // check if event is not recorded yet
  if (_event.signaledValue >= _signalCounter) {
    return false;
  }

  [_event notifyListener:_listener atValue:_signalCounter block:block];
  if (syncEvent) {
    _stream->synchronize(SyncType::COMMIT);
  }
  return true;
}

void MPSEvent::recordEvent(bool needsLock, bool syncEvent) {
  if (!needsLock) {
    recordEventLocked(syncEvent);
    return;
  }
  dispatch_sync(_stream->queue(), ^() {
    @autoreleasepool {
      recordEventLocked(syncEvent);
    }
  });
}

bool MPSEvent::waitForEvent(bool needsLock, bool syncEvent) {
  if (!needsLock) {
    return waitForEventLocked(syncEvent);
  }
  __block bool waited = false;

  dispatch_sync(_stream->queue(), ^() {
    @autoreleasepool {
      waited = waitForEventLocked(syncEvent);
    }
  });
  return waited;
}

bool MPSEvent::notifyEvent(bool needsLock, bool syncEvent, MTLSharedEventNotificationBlock block) {
  if (!needsLock) {
    return notifyEventLocked(syncEvent, block);
  }
  __block bool scheduledNotify = false;
  dispatch_sync(_stream->queue(), ^() {
    @autoreleasepool {
      scheduledNotify = notifyEventLocked(syncEvent, block);
    }
  });
  return scheduledNotify;
}

bool MPSEvent::queryEvent() const {
  // return false if not recorded or signaled yet
  return _signalCounter && (_event.signaledValue >= _signalCounter);
}

} // namespace mps
} // namespace at
