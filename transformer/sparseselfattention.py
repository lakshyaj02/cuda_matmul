import torch.nn as nn
from torch.nn.functional import *
import torch
from collections import namedtuple

from torch.autograd.function import InplaceFunction

import torch.sparse as sparse

import custom_mm
from matmuls.linear import Linear

from matmuls.matmuls import cusparseMM
from blocksparse.matmuls import blocksparseMM


custom_mm.init_cusparse()

def multi_head_attention_forward(query,                           # type: torch.Tensor
                                 key,                             # type: torch.Tensor
                                 value,                           # type: torch.Tensor
                                 embed_dim_to_check,              # type: int
                                 num_heads,                       # type: int
                                 in_proj_weight,                  # type: torch.Tensor
                                 in_proj_bias,                    # type: torch.Tensor
                                 bias_k,                          # type: torch.Tensor (optional)
                                 bias_v,                          # type: torch.Tensor (optional)
                                 add_zero_attn,                   # type: bool
                                 dropout_p,                       # type: float
                                 out_proj_weight,                 # type: torch.Tensor (optional)
                                 out_proj_bias,                   # type: torch.Tensor (optional)
                                 training=True,                   # type: bool
                                 key_padding_mask=None,           # type: torch.Tensor (optional)
                                 need_weights=True,               # type: bool
                                 attn_mask=None,                  # type: torch.Tensor (optional)
                                 use_separate_proj_weight=False,  # type: bool
                                 q_proj_weight=None,              # type: torch.Tensor (optional)
                                 k_proj_weight=None,              # type: torch.Tensor (optional)
                                 v_proj_weight=None,              # type: torch.Tensor (optional)
                                 static_k=None,                   # type: torch.Tensor (optional)
                                 static_v=None,                   # type: torch.Tensor (optional)
                                 key_padding_mask_mode='add',     # type: String ['add', 'mul']
                                 attn_mask_mode='add'             # type: String ['add', 'mul']
                                 ):

    bsz, tgt_len, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    assert key.size() == value.size()

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5


    if not use_separate_proj_weight:
        if torch.equal(query, key) and torch.equal(key, value):
            # self-attention
            q, k, v = cusparseMM.apply(query, in_proj_weight).chunk(3, dim=-1) + in_proj_bias.chunk(3, dim=-1)
            # q, k, v = Linear.apply(query, in_proj_weight).chunk(3, dim=-1) + in_proj_bias.chunk(3, dim=-1)
            # q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        elif torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            
            q = cusparseMM.apply(query, _w) + _b
            # q = Linear.apply(query, _w) + _b
            # q = linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = cusparseMM.apply(key, _w).chunk(2, dim=-1) + _b.chunk(2, dim=-1)
                # k, v = Linear.apply(key, _w).chunk(2, dim=-1) + _b.chunk(2, dim=-1)
                # k, v = linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = cusparseMM.apply(query, _w) + _b
            # q = Linear.apply(query, _w) + _b
            # q = linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = cusparseMM.apply(key, _w) + _b
            # k = Linear.apply(key, _w) + _b
            # k = linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = cusparseMM.apply(value, _w) + _b
            # v = Linear.apply(value, _w) + _b
            # v = linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = cusparseMM.apply(query, q_proj_weight_non_opt) + in_proj_bias[0:embed_dim]
            # q = Linear(query, q_proj_weight_non_opt) + in_proj_bias[0:embed_dim]
            # q = linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = cusparseMM.apply(key, k_proj_weight_non_opt) + in_proj_bias[embed_dim:(embed_dim * 2)]
            # k = Linear(key, k_proj_weight_non_opt) + in_proj_bias[embed_dim:(embed_dim * 2)]
            # k = linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:(embed_dim * 2)])
            v = cusparseMM.apply(value, v_proj_weight_non_opt) + in_proj_bias[(embed_dim * 2):]
            # v = Linear(value, v_proj_weight_non_opt) + in_proj_bias[(embed_dim * 2):]
            # v = linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2):])
        else:
            q = cusparseMM.apply(query, q_proj_weight_non_opt) + in_proj_bias
            # q = Linear(query, q_proj_weight_non_opt) + in_proj_bias
            # q = linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = cusparseMM.apply(key, k_proj_weight_non_opt) + in_proj_bias
            # k = Linear(key, k_proj_weight_non_opt) + in_proj_bias
            # k = linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = cusparseMM.apply(value, v_proj_weight_non_opt) + in_proj_bias
            # v = Linear(value, v_proj_weight_non_opt) + in_proj_bias
            # v = linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask,
                                      torch.zeros((attn_mask.size(0), 1),
                                                  dtype=attn_mask.dtype,
                                                  device=attn_mask.device)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros((key_padding_mask.size(0), 1),
                                                   dtype=key_padding_mask.dtype,
                                                   device=key_padding_mask.device)], dim=1)
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)

    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if attn_mask is not None:
        assert attn_mask.size(0) == src_len
        assert attn_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = torch.cat([attn_mask, torch.zeros((attn_mask.size(0), 1),
                                                          dtype=attn_mask.dtype,
                                                          device=attn_mask.device)], dim=1)
        if key_padding_mask is not None:
            key_padding_mask = torch.cat(
                [key_padding_mask, torch.zeros((key_padding_mask.size(0), 1),
                                                dtype=key_padding_mask.dtype,
                                                device=key_padding_mask.device)], dim=1)

    q = q.view(bsz, num_heads, q.shape[1], q.shape[2])
    k = k.view(bsz, num_heads, k.shape[1], k.shape[2])
    v = v.view(bsz, num_heads, v.shape[1], v.shape[2])
    # attention scores
    attn_output_weights = cusparseMM.apply(q, k.transpose(2,3))
    # attn_output_weights = sparse_dot_sdd_nt(q, k)
    print(attn_output_weights.shape)
    print(key_padding_mask.shape)
    masked_attn_output_weights = attn_output_weights + key_padding_mask.unsqueeze(1).unsqueeze(2)
    attn_output_weights = sparse.softmax(masked_attn_output_weights, dim=-1)
    # attn_output_weights = sparse_softmax(attn_output_weights, key_padding_mask=key_padding_mask, attn_mask=attn_mask, key_padding_mask_mode=key_padding_mask_mode, attn_mask_mode=attn_mask_mode)
    # outputs
    attn_output = cusparseMM.apply(attn_output_weights, v)
    # attn_output = sparse_dot_dsd_nn(attn_output_weights, v)
    attn_output = attn_output.view(bsz*num_heads, attn_output.shape[2], attn_output.shape[3])
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = cusparseMM.apply(attn_output, out_proj_weight) + out_proj_bias
    # attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
    return attn_output, None



class MultiheadAttention(nn.modules.activation.MultiheadAttention):

    # constructor
    def __init__(self, embed_dim, num_heads, blocksize=4, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None,
                 key_padding_mask_mode='add', attn_mask_mode='mul'):
        
        super(MultiheadAttention, self).__init__(embed_dim, num_heads, blocksize, bias, add_bias_kv, add_zero_attn, kdim, vdim)
        self.key_padding_mask_mode = key_padding_mask_mode
        self.attn_mask_mode = attn_mask_mode
        self.training = True


    # forward pass
    def forward(self, query, key, value, key_padding_mask=None,
                need_weights=True, attn_mask=None):
        # check that operation is supported
        if query.shape != key.shape or key.shape != value.shape:
            raise NotImplementedError('only self-attention is supported for now')

        # execute
        if not self._qkv_same_embed_dim:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                key_padding_mask_mode=self.key_padding_mask_mode, attn_mask_mode=self.attn_mask_mode)
        else:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask,
                key_padding_mask_mode=self.key_padding_mask_mode, attn_mask_mode=self.attn_mask_mode)

def replace_mha(model, info):
    for child_name, child in model.named_children():
        if isinstance(child, nn.modules.activation.MultiheadAttention):
            add_bias_kv = child.bias_k is not None
            device = child.in_proj_weight.device
            mha = MultiheadAttention(child.embed_dim, child.num_heads, info,
                                     dropout=child.dropout, add_bias_kv=add_bias_kv, 
                                     add_zero_attn=child.add_zero_attn, kdim=child.kdim,
                                     vdim=child.vdim, key_padding_mask_mode='mul', 
                                     attn_mask_mode='add').to(device)
            for yparam, xparam in zip(mha.parameters(), child.parameters()):
                yparam.data.copy_(xparam.data)
            setattr(model, child_name, mha)
        else:
            replace_mha(child, info)

custom_mm.destroy_cusparse()
