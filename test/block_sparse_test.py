import sys 
import os
sys.path.append(os.path.abspath("/home/lj9979/cuda_matmul"))

import torch
import torch.nn as nn
from torch.nn.functional import *
import torch
import numpy as np

import custom_mm
from blocksparse.matmuls import blocksparseMM
custom_mm.init_blocksparse()
def test_op():
    torch.manual_seed(0)

    SeqLen, Embed = 8, 8
    block_size = 2

    # query      = torch.rand(SeqLen, Embed).cuda()
    # key        = torch.rand(SeqLen, Embed).cuda()

    # query      = torch.rand(SeqLen, Embed)
    # key        = torch.rand(SeqLen, Embed)

    query = torch.tensor([[1.,2.,0.,0.],[3.,4.,0.,0.],[5.,6.,0.,0.],[7.,8.,0.,0.]])
    key = torch.tensor([[1.,5.,9.],[2.,6.,10.],[3.,7.,11.],[4.,8.,12.]])

    # query = torch.rand(32,32)
    # key = torch.rand(32,32)

    exp = query@key

    # q_cu = cusparseMM.apply(query, key)
    q = blocksparseMM.apply(query, key, block_size)

    print(query)
    print(key)
    print(q)
    print(exp)
    

    if torch.allclose(exp.to(device='cuda'), q, atol=1e-5):
        print(f'\nTest passed!\n')

if __name__ == "__main__":
    test_op()

custom_mm.destroy_blocksparse()