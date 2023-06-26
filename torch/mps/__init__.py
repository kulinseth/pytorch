r"""
This package enables an interface for accessing MPS backend in python
"""
import torch
from .. import Tensor

_is_in_bad_fork = getattr(torch._C, "_mps_is_in_bad_fork", lambda: False)
_default_mps_generator: torch._C.Generator = None  # type: ignore[assignment]

# local helper function (not public or exported)
def _get_default_mps_generator() -> torch._C.Generator:
    global _default_mps_generator
    if _default_mps_generator is None:
        _default_mps_generator = torch._C._mps_get_default_generator()
    return _default_mps_generator

def synchronize() -> None:
    r"""Waits for all kernels in all streams on a MPS device to complete."""
    return torch._C._mps_synchronize()

def get_rng_state() -> Tensor:
    r"""Returns the random number generator state as a ByteTensor."""
    return _get_default_mps_generator().get_state()

def set_rng_state(new_state: Tensor) -> None:
    r"""Sets the random number generator state.

    Args:
        new_state (torch.ByteTensor): The desired state
    """
    new_state_copy = new_state.clone(memory_format=torch.contiguous_format)
    _get_default_mps_generator().set_state(new_state_copy)

def manual_seed(seed: int) -> None:
    r"""Sets the seed for generating random numbers.

    Args:
        seed (int): The desired seed.
    """
    # the torch.mps.manual_seed() can be called from the global
    # torch.manual_seed() in torch/random.py. So we need to make
    # sure mps is available (otherwise we just return without
    # erroring out)
    if not torch.has_mps:
        return
    seed = int(seed)
    _get_default_mps_generator().manual_seed(seed)

def seed() -> None:
    r"""Sets the seed for generating random numbers to a random number."""
    _get_default_mps_generator().seed()

from . import profiler
from .event import Event
from .memory import *

__all__ = [
    'get_rng_state', 'manual_seed', 'seed', 'set_rng_state', 'synchronize',
    'empty_cache', 'set_per_process_memory_fraction', 'current_allocated_memory',
    'driver_allocated_memory', 'set_allocator_settings', 'memory', 'profiler', 'Event']
