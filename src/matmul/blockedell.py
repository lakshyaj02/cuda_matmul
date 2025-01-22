import torch
import numpy

'''
Class to store and hold all the variables associated with
the BlockedEll Format
'''

class BlockedEllMatrixBase(torch.nn.Module):
    def __init__(self, 
                shape, 
                block_mask, 
                # data,
                dense_data, 
                sparse_data, 
                block_shape=(32, 32)):
        super(BlockedEllMatrixBase, self).__init__()
        self.int_type = torch.int32

        if len(shape) != 2:
            raise Exception("shape should be a tuple of 2 ints")

        self.shape = torch.Size(shape)
        if len(block_shape) != 2:
            raise Exception("block_shape should be a tuple of 2 ints")

        self.block_shape = tuple(block_shape)

        self.block_mask = block_mask

        self.dense_data = torch.nn.Parameter(dense_data)

        self.sparse_data = torch.nn.Parameter(sparse_data)

        self.rebuild(dense_data, block_mask, callback=False)

    def get_sparse_differentiable_data(self):
        return self.sparse_data
    
    def get_dense_differentiable_data(self):
        return self.dense_data
    
    def get_dense_data(self):
        return self.dense_data
    
    @staticmethod
    def blocks_count_(shape, block_shape):
        return torch.Size((shape[0] // block_shape[0], shape[1] // block_shape[1]))
    
    @staticmethod
    def rand_block_mask(self, dense, block_shape=(32,32)):
        X, Y = self.blocks_count_(dense.shape, block_shape)
        rand_frac = numpy.random.rand(1)[0]
        positions = numpy.random.choice(X * Y, size=1+int(X*Y*rand_frac), replace=False)
        positions = torch.tensor(positions, dtype=torch.int64).sort()[0]
        block_mask = torch.zeros(X * Y, dtype=torch.bool)
        block_mask[positions] = True
        block_mask = block_mask.view(X, Y)
    
    def build_col_indices(self, data, block_mask, block_shape):
        ell_values_ = []
        temp_ell_cols = torch.zeros((int(block_mask.shape[0]), int(block_mask.shape[1])))

        for i in range(int(block_mask.shape[0])):
            ell_values_.append([])
            for j in range(int(block_mask.shape[1])):
                if block_mask[i,j] != 0:
                    temp_ell_cols[i, j] = j
                    non_zero_count_row += 1
                    ell_values_[i].append(data.weight[i*block_shape[0]:(i+1)*block_shape[0], j*block_shape[1]:(j+1)*block_shape[1]].squeeze(-1))
                else:
                    temp_ell_cols[i, j] = -1

            non_zero = max(non_zero, non_zero_count_row)
            non_zero_count_row = 0

        ell_cols = torch.zeros((int(block_mask.shape[0]) , int(non_zero)))

        # Get the column indices of the non-zero blocks
        for i in range(int(block_mask.shape[0])):
            count = 0
            non_zero_row = len(ell_values_[i])
            for j in range(int(block_mask.shape[1])):
                if temp_ell_cols[i, j] == j and count < non_zero:
                    ell_cols[i, count] = j
                    count += 1
                if count >= non_zero_row and count < non_zero:
                    ell_cols[i, count] = -1
                    count += 1
            count = 0

        a_ell_cols = int(ell_cols.shape[-1]*block_shape[1])

        return a_ell_cols

    def rebuild(self, data, block_mask, callback=True):
        if len(data.shape) != 2:
            raise Exception("data should be bidimensional, not of shape %s" % data.shape)
        if data.shape[0] != block_mask.shape[0] * self.block_shape[0]:
            raise Exception(
                "data.shape[0] (%d) should be equal to block_mask.shape[0]*block_shape[0] (%d)"
                % (data.shape[0], block_mask.shape[0] * self.block_shape[0])
            )
        if data.shape[1] != block_mask.shape[1] * self.block_shape[1]:
            raise Exception(
                "data.shape[1] (%d) should be equal to block_mask.shape[1]*block_shape[1] (%d)" % (data.shape[1], block_mask.shape[1] * self.block_shape[1])
            )
        if data.dtype != torch.float32 and data.dtype != torch.float16:
            raise Exception("data should be float32 or float16, not of type %s" % data.dtype)
        
    def sanity_check(self, dense, dense_weight, sparse_weight, block_mask):
        if self.block_count(block_mask) == 0:
            raise Exception("No outlier blocks are enabled in the block mask")
        if not torch.allclose(dense, dense_weight+sparse_weight):
            raise Exception("dense and dense_weight are not equal")

    def block_count(self, block_mask):
        return torch.sum(block_mask).item()

    @classmethod
    def zeros(cls, shape, block_shape=(32,32), block_mask=None, device=torch.device("cuda")):
        for i in range(2):
            if shape[i] % block_shape[i] != 0:
                raise Exception(f"Invalid shape: shape[{i}]={shape[i]} %% block_shape[{i}]={block_shape[i]} is not 0.")

        dense_data = torch.zeros(
            shape,
            dtype=torch.float16,
            device=device,
        )

        sparse_data = torch.zeros(
            shape,
            dtype=torch.float16,
            device=device,
        )

        return cls(
            shape, 
            block_mask, 
            dense_data, 
            sparse_data, 
            block_shape=(32, 32)
        )

    @classmethod
    def from_randn(cls, shape, block_shape=(32,32), block_mask=None, device=torch.device("cuda")):
        for i in range(2):
            if shape[i] % block_shape[i] != 0:
                raise Exception(f"Invalid shape: shape[{i}]={shape[i]} %% block_shape[{i}]={block_shape[i]} is not 0.")

        dense_data = torch.randn(
            shape,
            dtype=torch.float16,
            device=device,
        )

        sparse_data = torch.randn(
            shape,
            dtype=torch.float16,
            device=device,
        )

        return cls(
            shape, 
            block_mask, 
            dense_data, 
            sparse_data, 
            block_shape=(32, 32)
        )


    @classmethod
    def from_dense(cls, dense, block_shape=(32,32), block_mask=None):
        if block_mask is None:
            block_mask = cls.rand_block_mask(dense, block_shape)
        
        ret = cls.zeros(dense.shape, block_mask, block_shape, dense.device)

        dense_weight = dense * ret.get_masking_weight(device = dense.device)
        sparse_weight = dense * ret.get_complement_masking_weight(device = dense.device)

        ret.sanity_check(dense, dense_weight, sparse_weight, block_mask)

        # Find the column indices using the block mask
        # ell_cols = ret.build_col_indices(dense_weight, block_mask, block_shape)

        ret.dense_data.copy_(dense_weight)
        ret.sparse_data.copy_(sparse_weight)

        del dense_weight
        del sparse_weight

        torch.cuda.empty_cache()

        return ret
    
    def update_data(self):
        dense_data = self.dense_data * self.get_masking_weight(device = dense_data.device)
        self.dense_data.copy_(dense_data)
        sparse_data = self.dense_data * self.get_complement_masking_weight(device = dense_data.device)
        self.sparse_data.copy_(sparse_data)

        del dense_data
        del sparse_data
        torch.cuda.empty_cache()


    def get_masking_weight(self, device):
        mask_indices = torch.argwhere(~self.block_mask).to(dtype=torch.long)
        dense_num_blocks = torch.sum(~self.block_mask).item()
        masking_weight = torch.zeros(self.shape, dtype=torch.float, device = device)

        for i in range(dense_num_blocks):
            xs = torch.arange(mask_indices[i,0].item()*self.block_shape[0], (mask_indices[i,0].item()+1)*self.block_shape[0])
            ys = torch.arange(mask_indices[i,1].item()*self.block_shape[1], (mask_indices[i,1].item()+1)*self.block_shape[1])
            x, y = torch.meshgrid(xs, ys, indexing='xy')
            x = x.flatten()
            y = y.flatten()
            masking_weight[x,y] = torch.ones(self.block_shape[0]*self.block_shape[1], dtype=torch.float, device = device)

        del mask_indices

        torch.cuda.empty_cache()

        return masking_weight
    
    def get_complement_masking_weight(self, device = "cuda"):
        mask_indices = torch.argwhere(self.block_mask).to(dtype=torch.long)
        dense_num_blocks = torch.sum(self.block_mask).item()
        masking_weight = torch.zeros(self.shape, dtype=torch.float, device = device)

        # 1's in sparse mask => weights are chosen => 1's on dense mask should be retained
        # move to cuda for computation
        
        for i in range(dense_num_blocks):
            xs = torch.arange(mask_indices[i,0].item()*self.block_shape[0], (mask_indices[i,0].item()+1)*self.block_shape[0])
            ys = torch.arange(mask_indices[i,1].item()*self.block_shape[1], (mask_indices[i,1].item()+1)*self.block_shape[1])
            x, y = torch.meshgrid(xs, ys, indexing='xy')
            x = x.flatten()
            y = y.flatten()
            masking_weight[x,y] = torch.ones(self.block_shape[0]*self.block_shape[1], dtype=torch.float, device = device)

        del mask_indices

        torch.cuda.empty_cache()

        return masking_weight