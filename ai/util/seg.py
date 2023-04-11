import torch


def fhbc_seg_to_fh_mask(seg):
    return torch.bitwise_or(
        seg == 0, # face
        seg == 1, # hair
    ).float()
