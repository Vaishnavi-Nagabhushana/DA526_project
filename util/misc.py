import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
from typing import Optional, List


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[torch.Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        tensor = self.tensors.to(device)
        mask = self.mask.to(device) if self.mask is not None else None
        return NestedTensor(tensor, mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[torch.Tensor]):
    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device

        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)

        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError('not supported')

    return NestedTensor(tensor, mask)


def _max_by_axis(the_list):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def accuracy(output, target, topk=(1,)):
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    return nn.functional.interpolate(input, size=size,
                                     scale_factor=scale_factor,
                                     mode=mode, align_corners=align_corners)


# Added back minimal distributed helper (explicitly required by StyTR.py)
def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    if is_dist_avail_and_initialized():
        return dist.get_world_size()
    return 1


def get_rank():
    if is_dist_avail_and_initialized():
        return dist.get_rank()
    return 0


def is_main_process():
    return get_rank() == 0
