import torch

__all__ = ["empty_cache", "current_allocated_memory", "driver_allocated_memory",
           "set_per_process_memory_fraction", "set_allocator_settings"]

def empty_cache() -> None:
    r"""Releases all unoccupied cached memory currently held by the caching
    allocator so that those can be used in other GPU applications.
    """
    torch._C._mps_emptyCache()

def current_allocated_memory() -> int:
    r"""Returns the current GPU memory occupied by tensors in bytes.

     .. note::
        The returned size does not include cached allocations in
        memory pools of MPSAllocator.
    """
    return torch._C._mps_currentAllocatedMemory()

def driver_allocated_memory() -> int:
    r"""Returns total GPU memory allocated by Metal driver for the process in bytes.

     .. note::
        The returned size includes cached allocations in MPSAllocator pools
        as well as allocations from MPS/MPSGraph frameworks.
    """
    return torch._C._mps_driverAllocatedMemory()

def set_per_process_memory_fraction(fraction) -> None:
    r"""Set memory fraction for limiting process's memory allocation on MPS device.
    The allowed value equals the fraction multiplied by recommended maximum device memory
    (obtained from Metal API device.recommendedMaxWorkingSetSize).
    If trying to allocate more than the allowed value in a process, it will raise an out of
    memory error in allocator.

    Args:
        fraction(float): Range: 0~2. Allowed memory equals total_memory * fraction.

    .. note::
       Passing 0 to fraction means unlimited allocations
       (may cause system failure if out of memory).
       Passing fraction greater than 1.0 allows limits beyond the value
       returned from device.recommendedMaxWorkingSetSize.
    """

    if not isinstance(fraction, float):
        raise TypeError('Invalid type for fraction argument, must be `float`')
    if fraction < 0 or fraction > 2:
        raise ValueError('Invalid fraction value: {}. Allowed range: 0~2'.format(fraction))

    torch._C._mps_setMemoryFraction(fraction)

def set_allocator_settings(settings: str):
    r"""Set memory allocator settings for the MPS device.

    Args:
        settings(str): a string containing one or a combination of comma-separated allocator
            configurations with format: "config:value,config:value,...".
            Here are the supported configurations:
            - "debug_verbosity": bit-flag integer value to enable printing allocator debug messages to console
            - "high_watermark_ratio": float value as a ratio of the device's total GPU memory
                to enable a hard limit for the total allowed allocations (0.0 disables the limit).
            - "low_watermark_ratio": float value as a ratio of the device's total GPU memory
                to enable a soft limit to attempt limiting memory allocations below the
                lower watermark level (0.0 disables the limit).
            - "small_heap_divisor": integer divisor to determine the size of small heap.
            - "large_heap_divisor": integer divisor to determine the size of large heap.
            - "xlarge_heap_divisor": integer divisor to determine the size of extra large heap.
            - "max_pow2_roundup_size_mb": largest size in Mega bytes that gets round up to the next
                power of 2 (0 disables rounding).

    .. note::
        Heap sizes are computed from dividing the recommendedMaxWorkingSetSize by the values of heap divisors.
        See :ref:`mps-memory-management` for more details about GPU memory management.
    """
    if not isinstance(settings, str):
        raise TypeError('Invalid type for `settings` argument, must be `str`')

    return torch._C._mps_setAllocatorSettings(settings)
