#!/usr/bin/env python3
import numpy as np
import cv2
from ai_saas.ai.cna import crop_and_align, unalign_coords


# DI_K_192 = 16
DI_K_192 = 2
OUTER_BOUNDARY_DIV = 32


def get_di_k(imsize):
    return int(DI_K_192 * (imsize / 192))


def get_dilate_kernel(imsize):
    di_k = get_di_k(imsize)
    return cv2.getStructuringElement(cv2.MORPH_RECT, (di_k, di_k))


def get_outer_boundary_size(imsize):
    return imsize // OUTER_BOUNDARY_DIV
