from functools import reduce

import torch
from torch import Tensor


def weight_softmax(x: Tensor, weight: Tensor):
    """

    :param x: [..., tags_num]
    :param weight: [..., tags_num]
    :return: pros: [..., tags_num]
    """
    x = x + weight
    x = torch.exp(x - torch.max(x, dim=-1, keepdim=True)[0])
    return x / torch.sum(x, dim=-1, keepdim=True)


def distribute(locations: Tensor, mask: Tensor, centers: Tensor, factors: Tensor):
    """
    discrete clamp normal distribution

    :param locations: [..., location_num, 2(start&end)]
    :param mask: [..., location_num], True means not masked.
    :param centers: [..., queries_num, 2(start&end)]
    :param factors: [..., queries_num, 4]
    :return: probabilities: [..., queries_num, location_num]
    """
    rel_locs = (locations.unsqueeze(-3) - centers.unsqueeze(-2)).unsqueeze(-1)
    dis = -rel_locs.transpose(-1, -2) @ factors.unsqueeze(-3) @ rel_locs
    return torch.masked_fill(dis.view(dis.shape[:-2]), ~mask.unsqueeze(-2), 0)


def flatten(loc: Tensor) -> Tensor:
    """
    flatten the start&end format coordinate into linear format
    :param loc: [..., 2(start&end)]
    :return: [...]
    """

    return torch.div((1 + loc[..., 1]) * loc[..., 1], 2, rounding_mode='floor') + loc[..., 0]


def relocate(loc: Tensor, length: Tensor) -> Tensor:
    """
    relocate linear coordinate into the start&end form
    :param loc: [proposals_num]
    :param length: sentence length
    :return: [proposals_num, 2(start&end)]
    """

    return get_table(length)[loc]


def get_table(length: Tensor) -> Tensor:
    """
    0,0 0,1 0,2 0,3
        1,1 1,2 1,3
            2,2 2,3
                3,3
    :param length: [1], tensor of single element for search of the s&e table
    :return: [(length + 1) * length / 2, 2]
    """

    start = reduce(lambda a, b: a + b, [list(range(i + 1)) for i in range(length)])
    end = reduce(lambda a, b: a + b, [[i] * (i + 1) for i in range(length)])
    return torch.as_tensor(list(map(lambda args: [*args], zip(start, end))), device=length.device)
