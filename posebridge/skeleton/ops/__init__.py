import os
import sys

import torch
from torch.autograd import Function
from torch.nn import Module, Parameter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import shift_cuda
except ImportError as e:
    raise ImportError("Build the Shift-GCN CUDA op first: "
                      "cd posebridge/skeleton/ops && python setup.py build_ext --inplace") from e


class ShiftFunction(Function):
    @staticmethod
    def forward(ctx, x, xpos, ypos, stride=1):
        x = x.contiguous()
        if stride != 1:
            ypos = ypos + 0.5
        out = shift_cuda.forward(x, xpos, ypos, stride)
        ctx.save_for_backward(x, out, xpos, ypos)
        ctx.stride = stride
        return out

    @staticmethod
    def backward(ctx, grad):
        x, out, xpos, ypos = ctx.saved_tensors
        gi, gx, gy = shift_cuda.backward(grad.contiguous(), x, out, xpos, ypos, ctx.stride)
        return gi, gx, gy, None


class Shift(Module):
    """Learnable temporal shift."""

    def __init__(self, channel, stride, init_scale=3):
        super().__init__()
        self.stride = stride
        self.xpos = Parameter(torch.zeros(channel))
        self.ypos = Parameter(torch.zeros(channel))
        self.xpos.data.uniform_(-1e-8, 1e-8)
        self.ypos.data.uniform_(-init_scale, init_scale)

    def forward(self, x):
        return ShiftFunction.apply(x, self.xpos, self.ypos, self.stride)
