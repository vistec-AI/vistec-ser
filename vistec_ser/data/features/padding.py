from typing import Callable, List

import torch


def pad_dup(x: torch.Tensor, max_len: int) -> torch.Tensor:
    """Pad an Arguments feature upto specified length.
    The Arguments is repeated until max_len is reached.
    """
    time_dim = x.shape[-1]
    tmp = x.clone()
    num_repeat = int(max_len / time_dim)
    remainder = max_len - num_repeat * time_dim
    x_rem = x[:, :remainder]
    for _ in range(num_repeat - 1):
        x = torch.cat([x, tmp], dim=-1)
    x_pad = torch.cat([x, x_rem], dim=-1)
    return x_pad


def pad_zero(x: torch.Tensor, max_len: int) -> torch.Tensor:
    """Pad Arguments feature up to specified length.
    The padded values are zero.
    Arguments
    """
    zeros = torch.zeros([x.shape[0], max_len - x.shape[1]])
    x_pad = torch.cat([x, zeros], dim=-1)
    return x_pad


def pad_X(X: List[torch.Tensor], pad_fn: Callable, max_len: int = None) -> torch.Tensor:
    """Pad a pack of array to a specified max_len.
    If max_len is not specified, longest preprocessing will
    be use as a max length. This function is used to
    pad the fbank features to its max_len, not chopped.
    Arguments
    """
    if not max_len:
        max_len = max([x.shape[-1] for x in X])
    out = [pad_fn(x, max_len) for x in X]
    return torch.stack(out)