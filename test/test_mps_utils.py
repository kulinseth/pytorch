# -*- coding: utf-8 -*-
# Owner(s): ["module: mps"]

import torch
from torch.utils._pytree import tree_map

import logging
import contextlib
import itertools

class LoggingTensor(torch.Tensor):
    elem: torch.Tensor

    __slots__ = ['elem']

    @staticmethod
    def __new__(cls, elem, *args, **kwargs):
        # The wrapping tensor (LoggingTensor) shouldn't hold any
        # memory for the class in question, but it should still
        # advertise the same device as before
        r = torch.Tensor._make_wrapper_subclass(
            cls, elem.size(),
            # TODO: clone strides and storage aliasing
            dtype=elem.dtype, layout=elem.layout,
            device=elem.device, requires_grad=elem.requires_grad
        )
        # ...the real tensor is held as an element on the tensor.
        r.elem = elem
        return r

    def __repr__(self):
        return f"LoggingTensor({self.elem})"

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        def unwrap(e):
            return e.elem if isinstance(e, LoggingTensor) else e

        def wrap(e):
            return LoggingTensor(e) if isinstance(e, torch.Tensor) else e

        rs = tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs)))
        logging.getLogger("LoggingTensor").info(f"{func.__module__}.{func.__name__}", args, kwargs, rs)
        return rs

# https://stackoverflow.com/questions/36408496/python-logging-handler-to-append-to-list
class LoggingTensorHandler(logging.Handler):

    def __init__(self, log_list) -> None:
        logging.Handler.__init__(self)
        self.log_list = log_list
        self.next_shortid = 0

    # WARNING: not deterministic over multiple threads, this matters for
    # autograd
    def _shortid(self, o: object) -> int:
        if not hasattr(o, '_shortid'):
            o._shortid = self.next_shortid
            self.next_shortid += 1
        return o._shortid

    def _fmt(self, a: object) -> str:
        if isinstance(a, LoggingTensor):
            return f'${self._shortid(a)}'
        elif isinstance(a, torch.nn.Parameter):
            return f'Parameter(..., size={tuple(a.size())})'
        elif isinstance(a, torch.Tensor):
            return f'Tensor(..., size={tuple(a.size())})'
        else:
            return repr(a)

    def emit(self, record):
        fmt_args = ", ".join(itertools.chain(
            (self._fmt(a) for a in record.args[0]),
            (f"{k}={self._fmt(v)}" for k, v in record.args[1].items())
        ))
        fmt_rets = ", ".join(self._fmt(a) for a in record.args[2]) \
            if isinstance(record.args[2], (list, tuple)) else self._fmt(record.args[2])
        self.log_list.append(f'{fmt_rets} = {record.msg}({fmt_args})')

@contextlib.contextmanager
def capture_logs():
    logger = logging.getLogger("LoggingTensor")
    log_list = []
    handler = LoggingTensorHandler(log_list)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        yield log_list
    finally:
        logger.removeHandler(handler)

def tracefunc(frame, event, arg, indent=None):
    if indent is None:
        indent = [0]
    if event == "call":
        indent[0] += 2
        print("-" * indent[0] + "> call function", frame.f_code.co_name)
    elif event == "return":
        print("<" + "-" * indent[0], "exit function", frame.f_code.co_name)
        indent[0] -= 2
    return tracefunc
