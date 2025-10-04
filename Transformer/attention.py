from typing import Optional
import torch
import torch.nn.functional as f

MY_INF = 1e12

def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              mask: Optional[torch.Tensor] = None):
    assert q.shape[-1] == k.shape[-1]
    d_k = k.shape[-1]
    tmp = torch.matmul(q, k.transpose(-2, -1)) / d_k**0.5
    if mask is not None:
        tmp.masked_fill_(mask, -MY_INF)
    tmp = f.softmax(tmp, -1)
    tmp = torch.matmul(tmp, v)
    return tmp