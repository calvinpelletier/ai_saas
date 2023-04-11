import math
import torch
import cv2
from ai_saas.ai.util.outer import get_outer_boundary_size


def union(*masks):
    return torch.clamp(sum(masks), min=0., max=1.)


def intersection(*masks):
    return math.prod(masks)


def dilate(mask, kernel):
    dilated_mask = cv2.dilate(mask.cpu().numpy() * 255., kernel)
    dilated_mask = torch.tensor(dilated_mask / 255.).to(mask.device)
    return (dilated_mask > 0.5).float()

def erode(mask, kernel):
    dilated_mask = cv2.erode(mask.cpu().numpy() * 255., kernel)
    dilated_mask = torch.tensor(dilated_mask / 255.).to(mask.device)
    return (dilated_mask > 0.5).float()


def inverse(mask):
    return 1. - mask


def get_outer_boundary_mask(imsize):
    outer_boundary_mask = torch.zeros(imsize, imsize)
    obs = get_outer_boundary_size(imsize)
    for y in range(imsize):
        for x in range(imsize):
            if x < obs or x >= imsize - obs or \
                    y < obs or y >= imsize - obs:
                outer_boundary_mask[y][x] = 1.
    return outer_boundary_mask


def get_inner_mask(inner_imsize, outer_imsize, device='cuda'):
    inner_mask = torch.zeros(outer_imsize, outer_imsize, device=device)
    half_delta = (outer_imsize - inner_imsize) // 2
    for y in range(inner_imsize):
        for x in range(inner_imsize):
            inner_mask[y + half_delta][x + half_delta] = 1.
    return inner_mask


def dilate_mask(mask, dilate_kernel):
    dilated_mask = cv2.dilate(mask.cpu().numpy() * 255., dilate_kernel)
    dilated_mask = torch.tensor(dilated_mask / 255.).to('cuda')
    return (dilated_mask > 0.5).float()
