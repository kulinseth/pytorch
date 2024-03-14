//  Copyright © 2022 Apple Inc.

#pragma once

#include <ATen/mps/MPSAllocatorConfig.h>
#include <ATen/mps/MPSStream.h>
#include <ATen/mps/MPSEvent.h>
#include <cstdio>
#include <mutex>
#include <set>
#include <mach/vm_page_size.h>
#include <c10/util/flat_hash_map.h>

// this implementation is based on CUDACachingAllocator.
// It utilizes Metal Heaps to improve the performance with buffer allocation.
// Do not include this header. Use MPSAllocatorInterface.h instead.
// TODO: Unify the logic with CUDACachingAllocator and remove redundant code.
namespace at {
namespace mps {
namespace HeapAllocator {

// buffer pools could be customized with a combination of usage flags
enum UsageFlags : uint32_t {
  PRIVATE = 0,
  SMALL   = (1 << 0), // small heaps have sizes of kSmallHeap, and large ones kLargeHeap
  SHARED  = (1 << 1), // shared pools allocated on devices with unified memory; otherwise, private between host/device
  MANAGED = (1 << 2), // managed storage mode
  HAZARD  = (1 << 3), // enables Automatic Hazard Tracking for the resources allocated on the pool
  SCALAR  = (1 << 4), // used to import CPU scalar values to GPU and use them in MPS Stream
};

struct HeapBlock;

struct BufferBlock {
  id<MTLBuffer> buffer;
  void* cpu_ptr = nullptr; // stores the pointer to CPU mapping of a Shared MTLBuffer
  size_t size; // size after alignment
  size_t requested_size; // requested size (before alignment)
  // buffer shape is used for retrieving base of views in cached graphs
  std::vector<int64_t> shape;
  bool in_use = false;
  HeapBlock* heap;
  id_t buf_id;
  // counter to candidate least recently used buffers for garbage collection
  uint32_t gc_count = 0;
  uint32_t use_count = 0;
  // counter to assign unique ids to buffer blocks
  static uint64_t buffer_counter;
  // Metal events used to sync GPU/CPU operations on the shared-storage buffers
  MPSEventPtr event;

  BufferBlock(size_t Size, size_t RequestedSize = 0, const id<MTLBuffer> Buffer = nullptr,
              HeapBlock* Heap = nullptr) :
              buffer(Buffer), size(Size), requested_size(RequestedSize),
              heap(Heap), buf_id(Buffer ? ++buffer_counter : 0) { }

  static bool Comparator(const BufferBlock* a, const BufferBlock* b) {
    return (a->size != b->size) ? a->size < b->size : (uintptr_t)a->buffer < (uintptr_t)b->buffer;
  }
  static size_t alignUp(size_t Size, size_t Alignment) {
    assert(((Alignment - 1) & Alignment) == 0);
    return ((Size + Alignment - 1) & ~(Alignment - 1));
  }
  uint32_t retainCount() const { return [buffer retainCount]; }
};
typedef bool (*BufferComparison)(const BufferBlock*, const BufferBlock*);

struct BufferPool;
struct AllocParams {
  AllocParams(size_t Alloc_Size, size_t Requested_Size, BufferPool* Pool, const MPSAllocatorConfig& Configs) :
              search_key(Alloc_Size), pool(Pool), buffer_block(nullptr),
              requested_size(Requested_Size), configs(Configs) { }
  size_t size() const { return search_key.size; }

  BufferBlock search_key;
  BufferPool* pool;
  BufferBlock* buffer_block;
  size_t requested_size;
  const MPSAllocatorConfig& configs;
  // true if we exceed the low watermark limit. In this case
  // we apply strategies to relieve the pressure before allocation.
  bool has_memory_pressure = false;
  // true if we're allocating on a unified memory device
  bool has_unified_memory = true;
};

struct HeapBlock {
  id<MTLHeap> heap;
  struct { size_t total, available; } size;
  BufferPool* pool;
  unsigned int n_buffers;
  id_t heap_id;
  // indicates if we split this heap to sub-allocate 'several' buffers (otherwise single buffer)
  bool is_split;
  // counter to assign unique ids to heap blocks
  static uint64_t heap_counter;

  HeapBlock(size_t Size, const id<MTLHeap> Heap = nullptr, BufferPool *Pool = nullptr) :
            heap(Heap), size({.total = Size, .available = Size}), pool(Pool),
            n_buffers(0), heap_id(Heap ? ++heap_counter : 0), is_split(true) { }

  static MTLResourceOptions getOptions(uint32_t usage) {
    // TODO: check the caching performance of write-combined mode
    MTLResourceOptions options = MTLResourceCPUCacheModeDefaultCache;

    if (usage & UsageFlags::MANAGED)
      options |= MTLResourceStorageModeManaged;
    else if (usage & UsageFlags::SHARED)
      options |= MTLResourceStorageModeShared;
    else
      options |= MTLResourceStorageModePrivate;

    options |= (usage & UsageFlags::HAZARD) ? MTLResourceHazardTrackingModeTracked : MTLResourceHazardTrackingModeUntracked;

    return options;
  }

  static HeapBlock* createHeapBlock(AllocParams& params, id<MTLDevice> device, uint32_t usage) {
    HeapBlock *heapBlock = nullptr;
    bool is_split = true;
    const size_t size = params.size();
    MTLHeapDescriptor *d = [MTLHeapDescriptor new];
    if (d) {
      if (size <= kMaxSmallAlloc) {
        d.size = params.configs.smallHeapSize();
      } else if (size < kMinLargeAlloc) {
        d.size = params.configs.largeHeapSize();
      } else if (size < params.configs.xLargeHeapSize() / 2 && !params.has_memory_pressure) {
        d.size = params.configs.xLargeHeapSize();
      } else {
        d.size = kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
        is_split = false;
      }
      d.storageMode = (usage & UsageFlags::SHARED) ? MTLStorageModeShared : MTLStorageModePrivate;
      d.cpuCacheMode = MTLCPUCacheModeDefaultCache;
      // this automatically handles Metal buffer access synchronizations at the
      // cost of slightly lower performance.
      d.hazardTrackingMode = (usage & UsageFlags::HAZARD) ? MTLHazardTrackingModeTracked : MTLHazardTrackingModeUntracked;
      d.resourceOptions = getOptions(usage);
      d.type = MTLHeapTypeAutomatic;
      id<MTLHeap> heap = [device newHeapWithDescriptor: d];
      if (heap) {
        [heap setPurgeableState:MTLPurgeableStateNonVolatile];
        const size_t heap_size = heapAvailableSize(heap);
        heapBlock = new HeapBlock(heap_size, heap, params.pool);
        if (heapBlock) {
          heapBlock->is_split = is_split;
        }
      }
      [d release];
    }
    return heapBlock;
  }
  static bool Comparator(const HeapBlock* a, const HeapBlock* b) {
    return (a->size.available != b->size.available) ? a->size.available < b->size.available :
                                                      (uintptr_t)a->heap < (uintptr_t)b->heap;
  }
  static NSUInteger heapAvailableSize(id<MTLHeap> heap, size_t Alignment = vm_page_size) {
    return [heap maxAvailableSizeWithAlignment:Alignment];
  }
  NSUInteger Size() {
    return [heap size];
  }
  id<MTLBuffer> newMTLBuffer(size_t length, uint32_t usage) {
    id<MTLBuffer> buf = [heap newBufferWithLength:length options:getOptions(usage)];
    if (buf) {
      updateAvailableSize();
      n_buffers++;
    }
    return buf;
  }
  // returns the retainCount before releasing the buffer
  uint32_t releaseMTLBuffer(id<MTLBuffer>& buffer) {
    const uint32_t retainCount = [buffer retainCount];
    [buffer release];
    buffer = nil;
    updateAvailableSize();
    n_buffers--;
    return retainCount;
  }
  // returns the retainCount before releasing the heap
  uint32_t releaseMTLHeap() {
    const uint32_t retainCount = [heap retainCount];
    TORCH_INTERNAL_ASSERT(!n_buffers); // assert if heap isn't empty
    [heap setPurgeableState:MTLPurgeableStateEmpty];
    [heap release];
    heap = nil;
    size.available = 0;
    return retainCount;
  }
  uint32_t retainCount() const { return [heap retainCount]; }
  void updateAvailableSize() { size.available = heapAvailableSize(heap); }
};
typedef bool (*HeapComparison)(const HeapBlock*, const HeapBlock*);

struct BufferPool {
  enum class Kind {
    PRIVATE_SMALL,
    PRIVATE_LARGE,
    SHARED_SMALL,
    SHARED_LARGE,
    SCALAR, // small shared buffers to import scalar values into MPS stream
  };

  BufferPool(const id<MTLDevice> Device, uint32_t Usage) :
             device(Device), usage(Usage),
             heaps(HeapBlock::Comparator), available_buffers(BufferBlock::Comparator) { }

  const id<MTLDevice> device;
  // usage flags to customize the pool for various purposes (see UsageFlags enum)
  const uint32_t usage;
  // total number of buffers in the pool
  uint32_t n_buffers = 0;
  // total allocations size on this pool
  size_t allocated_size = 0;
  // total memory available in the pool
  size_t available_size = 0;
  // list of heaps ordered by their "available" (not total) memory size
  std::set<HeapBlock*, HeapComparison> heaps;
  // list of only "available" buffers in the pool (i.e., buffers not in-use)
  std::set<BufferBlock*, BufferComparison> available_buffers;
  // list of buffers that are in a state of "limbo" where they've already been freed
  // from PyTorch-side, but were not returned to pool due to still being
  // in-use by command buffers with retainCount > 1. In this state, the buffer is
  // neither ready to be recycled, nor could be returned to pool as available.
  // These buffers will be returned to pool once the command buffer's
  // completionHandler callbacks are called.
  std::unordered_set<BufferBlock*> buffers_pending_free;
  // list of heaps pending size update
  std::unordered_set<HeapBlock*> heaps_pending_update;
};

class MPSHeapAllocatorImpl {
public:
  explicit MPSHeapAllocatorImpl() :
    m_device(at::mps::MPSDevice::getInstance()->device()),
    m_max_buffer_size([m_device maxBufferLength]),
    m_configs([m_device hasUnifiedMemory], [m_device recommendedMaxWorkingSetSize]),
    m_stream(getDefaultMPSStream()),
    m_event_pool(getMPSEventPool()) {
    init_allocator();
  }
  ~MPSHeapAllocatorImpl() {
    emptyCache();
  }
  // interface exposed to at::Allocator
  id<MTLBuffer> malloc(size_t size, uint32_t usage);
  // frees a buffer and returns it into buffer pool
  void free(void* ptr);
  // releases all the cached buffers and their associated heaps
  void emptyCache();
  // free inactive buffers that are pending to be freed
  void freeInactiveBuffers();
  // returns true if buffer was allocated from the shared pool
  bool isSharedBuffer(void* ptr);
  // get the requested unaligned size of an MTLBuffer
  ssize_t getUnalignedBufferSize(void* ptr);
  // set the shape of a base tensor from a view tensor
  void setBufferShape(void* ptr, const IntArrayRef& shape);
  // retrieve the shape of a base tensor from a view tensor
  IntArrayRef getBufferShape(void* ptr);
  // get the unique ID of the buffer
  id_t getBufferId(void* ptr);
  // allocate a buffer from a specialized pool to import CPU scalars into GPU
  id<MTLBuffer> allocScalarBufferWithValue(void* value, size_t size);
  // returns a CPU-mapping of the input buffer and its retainCount,
  // if only it has Shared storage-mode and allocated on MPSAllocator
  std::pair<void*, uint32_t> getSharedBufferPtr(void* buffer);
  // records events for a list of MTLBuffers (list is used to lock the mutex once)
  // returns true if records any event (given if passed buffers exist and are shared-storage)
  bool recordEvents(c10::ArrayRef<void*> buffers);
  // waits for the event to signal the completion of GPU execution
  // on the passed shared buffers (list is used to lock the mutex once)
  // returns true if actually waited on any event
  bool waitForEvents(c10::ArrayRef<void*> buffers);
  // this indicates how far (in Megabytes) the current total allocations are from the
  // low watermark limit which is used to detect if we're under memory pressure
  // This returns zero if we've reached the low watermark limit
  ssize_t getLowWatermarkValue();
  // (see m_total_allocated_memory for description)
  size_t getTotalAllocatedMemory() const { return m_total_allocated_memory; }
  // (see m_current_allocated_memory for description)
  size_t getCurrentAllocatedMemory() const { return m_current_allocated_memory; }
  // total GPU memory allocated in the process by Metal driver; including
  // implicit allocations from MPS/MPSGraph frameworks and MPSHeapAllocatorImpl.
  size_t getDriverAllocatedMemory() const { return current_allocated_size(); }
  // returns the device that we allocate from
  inline id<MTLDevice> Device() const { return m_device; }
  // allocator configs
  MPSAllocatorConfig& configs() { return m_configs; }
  // TODO: make a common function to do size unit conversions in PyTorch.
  inline std::string format_size(uint64_t size) const { return m_configs.format_size(size); }

private:
  const id<MTLDevice> m_device;
  std::recursive_mutex m_mutex;
  // allocated buffers by device pointer
  ska::flat_hash_map<void*, BufferBlock*> m_allocated_buffers;
  // using a container for pools to simplify iterating them
  ska::flat_hash_map<BufferPool::Kind, std::unique_ptr<BufferPool>> m_pools;
  // total memory allocated by HeapAllocator (including blocks in pools)
  size_t m_total_allocated_memory = 0;
  // currently active memory allocations in use (i.e., blocks not in pools)
  size_t m_current_allocated_memory = 0;
  // max buffer size allowed by Metal
  size_t m_max_buffer_size = 0;
  // allocator configurations
  MPSAllocatorConfig m_configs;
  // default MPS stream
  MPSStream* m_stream;
  // we hold a reference to MPSEventPool so it could get destroyed after MPSAllocator
  std::shared_ptr<MPSEventPool> m_event_pool;

  void init_allocator();
  void init_buffer_pools();
  HeapBlock* get_free_heap(AllocParams& params);
  bool get_free_buffer(AllocParams& params);
  BufferBlock* get_allocated_buffer_block(void* ptr);
  BufferBlock* alloc_buffer_block(size_t size, uint32_t usage);
  bool alloc_buffer(AllocParams& params);
  void free_buffer(BufferBlock* buffer_block);
  // returns true if the container heap is also released
  bool release_buffer(BufferBlock* buffer_block, bool remove_empty_heap = true);
  void release_buffers(BufferPool& pool);
  bool release_available_cached_buffers(AllocParams& params);
  bool release_cached_buffers();
  // free unused cached blocks to reclaim GPU memory if memory pressure is high
  void garbage_collect_cached_buffers(AllocParams& params);
  // returns the suitable buffer pool type for the usage or
  // requested/allocated sizes
  BufferPool& get_pool(size_t requested_size, size_t aligned_size, uint32_t usage);
  // returns the aligned allocation size that is optimized
  // for the buffers to get reused frequently
  size_t get_allocation_size(size_t size, uint32_t usage) const;
  // maximum size of device memory available for allocation in current process
  // Note: the recommendedMaxWorkingSetSize is typically 75% of the total system memory.
  size_t max_device_size() const { return [m_device recommendedMaxWorkingSetSize]; }
  // there are implicit allocations from MPS backend, so we need to query the 'device' for
  // total allocated size instead of manually tracking in MPSAllocator
  size_t current_allocated_size() const { return [m_device currentAllocatedSize]; }

  bool trigger_memory_callbacks(BufferBlock* buffer_block, IMpsAllocatorCallback::EventType event) const {
    for (const auto& name : MPSAllocatorCallbacksRegistry()->Keys()) {
      MPSAllocatorCallbacksRegistry()->Create(name)->executeMPSAllocatorCallback(buffer_block ? buffer_block->buffer : nullptr, event);
    }
    return true;
  }
};

} // namespace HeapAllocator
} // namespace mps
} // namespace at
