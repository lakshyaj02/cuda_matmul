import torch
from .matmuls import blocksparseMM
import torch.nn as nn
import random
import numpy
from .blockedell import BlockedEllMatrixBase

class BlockedEllLinear(nn.Module):
    def __init__(self, in_features:int, out_features:int, torch_nn_linear=None, block_size=(16,16), block_mask=None):
        super(BlockedEllLinear, self).__init__()
        self.block_size = block_size
        self.fn = blocksparseMM.apply
        self.dense_fn = torch.nn.functional.linear
        
        if block_mask is None:
            X, Y = out_features // self.block_shape[0], in_features // self.block_shape[1]
            rand_frac = random.uniform(0.1, 1.0)
            # rand_frac = numpy.random.rand(1)[0]
            positions = numpy.random.choice(X * Y, size=1+int(X*Y*rand_frac), replace=False)
            positions = torch.tensor(positions, dtype=torch.int64, device=torch_nn_linear.weight.device).sort()[0]
            block_mask = torch.zeros(X * Y, dtype=torch.bool, device=torch_nn_linear.weight.device)
            block_mask[positions] = True
            block_mask = block_mask.view(X, Y)

        BlockedEllMatrixConstructor = BlockedEllMatrixBase

        if torch_nn_linear is not None:
            self.in_features = torch_nn_linear.in_features
            self.out_features = torch_nn_linear.out_features
            self.bias = torch_nn_linear.bias is not None

        if self.in_features % self.block_size[1] != 0:
            raise Exception(
                f"BlockedEllLinear invalid in_features={self.in_features}, should be multiple of {self.block_size[1]}"
            )
        if self.out_features % self.block_size[0] != 0:
            raise Exception(
                f"BlockedEllLinear invalid out_features={out_features}, should be multiple of {self.block_size[0]}"
            )
        
        if torch_nn_linear is not None:
            with torch.no_grad():
                self.blocked_ell_matrix = BlockedEllMatrixConstructor.from_dense(
                    torch_nn_linear.weight,
                    block_shape=self.block_size,
                    block_mask=block_mask,
                )

        else:
            self.blocked_ell_matrix = BlockedEllMatrixConstructor.from_randn(
                block_shape=self.block_size,
                block_mask=self.block_mask,
            )
        

    def forward(self, x):
        x = x.to(self.sparse_weight.get_dense_differentiable_data().dtype)
        if self.bias is not None:
            self.bias.data = self.bias.to(x.dtype)
        x_sparse = self.fn(self.sparse_weight.get_sparse_differentiable_data(), x, self.block_size)
        x_dense = self.dense_fn(x, self.sparse_weight.get_dense_differentiable_data(), self.bias)
        x = x_sparse + x_dense
        return x
        # return blocksparseMM.apply(w1, x, self.block_size)