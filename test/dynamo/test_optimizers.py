# Owner(s): ["module: dynamo"]

import contextlib
import inspect
import unittest

import torch

import torch._dynamo
import torch._dynamo.test_case
import torch._dynamo.testing


input = torch.ones([10, 10])
model = torch.nn.Sequential(*[torch.nn.Linear(10, 10) for _ in range(2)])
model(input).sum().backward()


# Include optimizer code for tracing
optim_filenames = set(
    [
        inspect.getfile(obj)
        for obj in torch.optim.__dict__.values()
        if inspect.isclass(obj)
    ]
)


optim_filenames |= {torch.optim._functional.__file__}


def make_test(optim_cls, exp_graph_count=1, closure=None, **kwargs):
    opt = optim_cls(model.parameters(), **kwargs)

    def test_fn(self):
        nonlocal opt
        if closure is not None:

            def fn():
                opt.step(closure)

        else:
            fn = opt.step

        _, _, graphs, _, _, _ = torch._dynamo.explain(fn)

        self.assertEqual(exp_graph_count, len(graphs))

    return test_fn


@contextlib.contextmanager
def enable_optimizer_tracing():
    try:
        old = set(torch._dynamo.skipfiles.FILENAME_ALLOWLIST)

        torch._dynamo.skipfiles.FILENAME_ALLOWLIST.update(optim_filenames)
        yield
    finally:
        torch._dynamo.skipfiles.FILENAME_ALLOWLIST.clear()
        torch._dynamo.skipfiles.FILENAME_ALLOWLIST.update(old)


class OptimizerTests(torch._dynamo.test_case.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        # needed until pytorch assertion is changed to enable Adam
        # to be called with capturable=True
        cls._exit_stack.enter_context(
            unittest.mock.patch.object(
                torch._dynamo.config, "capture_scalar_outputs", True
            )
        )
        cls._exit_stack.enter_context(enable_optimizer_tracing())

    test_sgd = make_test(torch.optim.SGD, lr=0.01)
    # lgbfs has data-dependent control and internally iterates
    # calling the closure
    # TODO mlazos: re-enable once we have latest pytorch with FakeTensor fix #497
    # test_lbfgs = make_test(
    #    torch.optim.LBFGS, exp_frame_cnt=3, closure=lambda: model(input).sum()
    # )

    # These optimizers are disabled until we remove item() calls
    test_adam = make_test(torch.optim.Adam, exp_graph_count=0)
    test_adamw = make_test(torch.optim.AdamW, exp_graph_count=0)

    # RAdam and Adagrad have data-dependent control which breaks the graph;
    # furthermore, the break is inside a for loop, so we bail on the frame
    # entirely.  This is basically an xfail; if the frame count goes up
    # you done good
    test_radam = make_test(torch.optim.RAdam, exp_graph_count=0)

    # ASGD has a small optimization that avoids averaging
    # This will fully capture the graph once that optimization is removed
    # test_asgd = make_test(torch.optim.ASGD, exp_graph_count=0)


# exclude SparseAdam because other areas of the stack don't support it yet
# the others are handled specially above
exclude = set(
    [
        "SGD",  # Handled above
        "ASGD",  # Disabled pending item call removal + optimization removal
        "Optimizer",
        "SparseAdam",  # Unsupported
        "LBFGS",  # Unsupported
        "Adam",  # Disabled pending item call removal
        "AdamW",  # Disabled pending item call removal
        "RAdam",  # Disabled pending item call removal
        "ASGD",
    ]
)

optimizers = [
    opt
    for opt in torch.optim.__dict__.values()
    if inspect.isclass(opt)
    and issubclass(opt, torch.optim.Optimizer)
    and opt.__name__ not in exclude
]


for opt in optimizers:
    setattr(OptimizerTests, "test_" + opt.__name__.lower(), make_test(opt))


class End2EndTests(torch._dynamo.test_case.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._exit_stack.enter_context(enable_optimizer_tracing())

    # https://github.com/pytorch/torchdynamo/issues/1604
    def test_optimizing_over_tensor_with_requires_grad(self):
        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                z = torch.bmm(x, y)
                z = torch.flatten(z, 1)
                return z

        def training_iter_fn(batch, model, optimizer):
            optimizer.zero_grad()
            out = model(**batch)
            target = torch.tensor([0, 7])
            loss = torch.nn.CrossEntropyLoss()(out, target)
            loss.backward()
            optimizer.step()
            return loss

        net = Net()
        input1 = torch.randn(2, 1, 4)
        input2 = torch.randn(2, 4, 8, requires_grad=True)
        optimizer = torch.optim.Adam([input2], lr=0.1)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_training_iter_fn = torch._dynamo.optimize(cnts)(training_iter_fn)
        batch = {"x": input1, "y": input2}
        for _ in range(2):
            opt_training_iter_fn(batch, net, optimizer)
        self.assertEqual(cnts.frame_count, 1)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
