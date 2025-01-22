import torch
from torch.autograd.function import InplaceFunction
import custom_mm

def get_sparse_tensor_properties(a: torch.Tensor):
    '''
    Retrieve properties of CSR tensor.
    :param a: CSR Tensor
    :returns: Row indices, col indices, values, number of nonzeros, and shape of a
    '''
    
    assert a.is_sparse_csr
    
    x = torch.Tensor.values(a).cuda(), torch.Tensor.col_indices(a).type(torch.IntTensor).cuda(), \
           torch.Tensor.crow_indices(a).type(torch.IntTensor).cuda(), len(torch.Tensor.values(a)), \
           a.shape[-2], a.shape[-1]
    return x

def get_blockell_tensor_properties(a: torch.Tensor, b: torch.Tensor, block_size:int = 8):
    '''
    Retrieve properties of blockell 2D sparse tensor.
    '''
    non_zero_count_row = 0
    non_zero = 0
    temp_ell_cols = torch.zeros((int(a.shape[0]/block_size), int(a.shape[1]/block_size)))

    ell_values_ = []

    # Get all the non-zero blocks in the matrix
    for i in range(int(a.shape[0]/block_size)):
        ell_values_.append([])
        for j in range(int(a.shape[1]/block_size)):
            if torch.nonzero(a[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size], as_tuple=False).numel() != 0:
                temp_ell_cols[i, j] = j
                non_zero_count_row += 1
                ell_values_[i].append(a[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size].squeeze(-1))
            else:
                temp_ell_cols[i, j] = -1

        non_zero = max(non_zero, non_zero_count_row)
        non_zero_count_row = 0

    ell_cols = torch.zeros((int(a.shape[0]/block_size) , int(non_zero)))

    # Get the column indices of the non-zero blocks
    for i in range(int(a.shape[0]/block_size)):
        count = 0
        non_zero_row = len(ell_values_[i])
        for j in range(int(a.shape[1]/block_size)):
            if temp_ell_cols[i, j] == j and count < non_zero:
                ell_cols[i, count] = j
                count += 1
            if count >= non_zero_row and count < non_zero:
                ell_cols[i, count] = -1
                count += 1
        count = 0

    a_ell_cols = int(ell_cols.shape[-1]*block_size)

    # Change b in a way it is compatible with the block ell representation
    b_col_maj = b.t().flatten()

    a_num_blocks = ell_cols.shape[0]*ell_cols.shape[1]

    # Finally return the block ell representation of the matrix
    # x = values_tensor, ell_cols.type(torch.IntTensor), a.shape[-2], a.shape[-1], block_size, a_ell_cols, a_num_blocks, b_col_maj
    x = a, ell_cols.type(torch.IntTensor), a.shape[-2], a.shape[-1], block_size, a_ell_cols, a_num_blocks, b_col_maj

    return x


def blocksparse_matmul(a: torch.Tensor,
                 b: torch.Tensor, block_size:int = 16,
                 mm_op=custom_mm.blocksparse_mmul) -> torch.Tensor:
    '''
    Uses a sparse kernel to perform matrix multiplication.

    :param a: This should be a blockell sparse tensor
    :param b:
    :param mm_op: kernel to perform basic matrix multiplication
    :returns: Matrix multiplication output
    '''
    a_shape = a.shape
    b_shape = b.shape
   
    c_rows = a_shape[0]
    c_cols = b_shape[-1]
    c = torch.zeros(tuple(list(a_shape[:-2]) + [c_rows, c_cols]), device=torch.device('cuda'))

    if len(a_shape) == 1 or len(b_shape) == 1:
        print('Matrix-vector multiplication is not implemented in cusparse blockedell')
        return a @ b

    # a_shape can't be 3 because csr tensor only supports 2d
    if len(a_shape) == 2 and len(b_shape) == 3:
        # flatten B into a 2d tensor
        ldb, dim1, dim2 = b_shape
        _b = b.reshape(dim1, ldb*dim2)
        c = torch.zeros(a.shape[0], ldb*dim2, device=torch.device('cuda'))
        c = mm_op(*get_blockell_tensor_properties(a, block_size), _b, c).reshape(ldb, -1, dim2)

    if len(a_shape) == 3 and len(b_shape) == 2:
        # flatten a into a 2d tensor
        lda, dim1, dim2 = a_shape
        c = torch.stack([naive_matmul(a[i], b, mm_op) for i in range(lda)])
            
    elif len(a_shape) >= 3 and len(b_shape) >= 3:
        lda, ldb = a_shape[0], b_shape[0]
        assert lda == ldb
        c = torch.stack([naive_matmul(a[i], b[i], mm_op) for i in range(lda)])

    elif len(a_shape) == 2 and len(b_shape) == 2:
        c = mm_op(*get_blockell_tensor_properties(a, b, block_size), b, c)
    else:
        print(
            'Multiplication with matrix dimensions is not implemented in cuSPARSE'
        )
        return a @ b
    return c
    
class blocksparseMM(InplaceFunction):
    @staticmethod
    def forward(ctx, m1, m2, block_size:int = 16):
        ctx.save_for_backward(m1, m2)
        return blocksparse_matmul(m1, m2, block_size)

    @staticmethod
    def backward(ctx, grad_output):
        m1, m2 = ctx.saved_variables
        grad_m1 = grad_m2 = None

        if ctx.needs_input_grad[0]:
            grad_m1 = blocksparse_matmul(grad_output, m2.transpose(
                -1, -2))
        if ctx.needs_input_grad[1]:
            grad_m2 = blocksparse_matmul(m1.transpose(
                -1, -2), grad_output)

        return grad_m1, grad_m2
    

def naive_matmul(a: torch.Tensor,
                 b: torch.Tensor,
                 mm_op=custom_mm.naive_spmm) -> torch.Tensor:
    '''
    Uses a sparse kernel to perform matrix multiplication.

    :param a: Torch CSR matrix
    :param b: 
    :param mm_op: kernel to perform basic matrix multiplication
    :returns: Matrix multiplication output
    '''
    a_shape = a.shape
    b_shape = b.shape

    c_rows = a_shape[-2]
    c_cols = b_shape[-1]
    c = torch.zeros(
        tuple(list(a_shape[:-2]) + [c_rows, c_cols]), device=torch.device('cuda'))

    if len(a_shape) == 1 or len(b_shape) == 1:
        print('Matrix-vector multiplication is not implemented in cuBLAS')
        return a @ b

    # a_shape can't be 3 because csr tensor only supports 2d
    if len(a_shape) == 2 and len(b_shape) == 3:
        if not a.is_sparse_csr:
            a = a.to_sparse_csr()
        # flatten B into a 2d tensor
        ldb, dim1, dim2 = b_shape
        _b = b.reshape(ldb * dim1, dim2)
        c = mm_op(*get_sparse_tensor_properties(a), _b, c).reshape(ldb, -1, dim2)
    


class naiveSpMM(InplaceFunction):
    @staticmethod
    def forward(ctx, m1, m2):
        # swap around for col-major call
        # where row major is expected
        ctx.save_for_backward(m1, m2)
        return naive_matmul(m1, m2)

    @staticmethod
    def backward(ctx, grad_output):
        m1, m2 = ctx.saved_variables
        grad_m1 = grad_m2 = None

        if ctx.needs_input_grad[0]:
            grad_m1 = naive_matmul(grad_output, m2.transpose(
                -1, -2))

        if ctx.needs_input_grad[1]:
            grad_m2 = naive_matmul(
                m1.transpose(-1, -2),
                grad_output)

        return grad_m1, grad_m2