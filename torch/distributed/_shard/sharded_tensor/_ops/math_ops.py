import torch
from torch import Tensor
from torch.distributed._shard.sharded_tensor import ShardedTensor, _sharded_op_impl


def binary_math_op_impl(op, types, args=(), kwargs=None, pg=None):
    """
    Handles ``__torch_function__`` dispatch for the binary math ops
    such as `torch.add`, `torch.mul`, `torch.div`, etc.
    This method computes on ShardedTensor, or ShardedTensor op
    """
    if len(args) != 2:
        raise ValueError("Only support binary math op on ShardedTensor for now!")
    lhs = args[0]
    rhs = args[1]
    # Validate types
    if isinstance(lhs, (int, float)):
        assert isinstance(rhs, ShardedTensor)
        res = op(lhs, rhs.local_tensor())
        return ShardedTensor._init_from_local_tensor(
            res,
            rhs.sharding_spec(),
            rhs.size(),  # type: ignore[arg-type]
            process_group=pg,
        )

    elif isinstance(rhs, (int, float)):
        assert isinstance(lhs, ShardedTensor)
        res = op(lhs.local_tensor(), rhs)
        return ShardedTensor._init_from_local_tensor(
            res,
            lhs.sharding_spec(),
            lhs.size(),  # type: ignore[arg-type]
            process_group=pg,
        )
    else:
        raise RuntimeError(
            f"torch function '{op.__name__}', with args: {args} and "
            f"kwargs: {kwargs} not supported yet for ShardedTensor!"
        )


def register_math_op(op):
    @_sharded_op_impl(op)
    def binary_math_op(types, args=(), kwargs=None, pg=None):
        return binary_math_op_impl(op, types, args, kwargs, pg)


binary_ops = [
    # add
    torch.add,
    Tensor.add,
    Tensor.__add__,
    Tensor.__radd__,
    # sub
    torch.sub,
    Tensor.sub,
    Tensor.__sub__,
    Tensor.__rsub__,
    # mul
    torch.mul,
    Tensor.mul,
    Tensor.__mul__,
    Tensor.__rmul__,
    # div
    torch.div,
    Tensor.div,
    Tensor.__div__,
    Tensor.__rdiv__,
]

for op in binary_ops:
    register_math_op(op)
