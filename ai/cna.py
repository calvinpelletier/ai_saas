import numpy as np
import scipy
import scipy.ndimage
from PIL import Image, ImageDraw
import math


ENABLE_PADDING = True
USE_PIXEL_CENTER = False
ALIGNED_MASK_BUFFER = 2


def crop_and_align(img, coords, output_size, transform_size):
    quad = np.reshape(np.asarray(coords) - 0.5, (-1, 2))
    x = (quad[3] - quad[0]) / 2
    qsize = np.hypot(*x) * 2

    # shrink
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (
            int(np.rint(float(img.size[0]) / shrink)),
            int(np.rint(float(img.size[1]) / shrink)),
        )
        img = img.resize(rsize, Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # crop
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (
        int(np.floor(min(quad[:, 0]))),
        int(np.floor(min(quad[:, 1]))),
        int(np.ceil(max(quad[:, 0]))),
        int(np.ceil(max(quad[:, 1]))),
    )
    crop = (
        max(crop[0] - border, 0),
        max(crop[1] - border, 0),
        min(crop[2] + border, img.size[0]),
        min(crop[3] + border, img.size[1]),
    )
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # pad
    pad = (
        int(np.floor(min(quad[:, 0]))),
        int(np.floor(min(quad[:, 1]))),
        int(np.ceil(max(quad[:, 0]))),
        int(np.ceil(max(quad[:, 1]))),
    )
    pad = (
        max(-pad[0] + border, 0),
        max(-pad[1] + border, 0),
        max(pad[2] - img.size[0] + border, 0),
        max(pad[3] - img.size[1] + border, 0),
    )
    if ENABLE_PADDING and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(
            np.float32(img),
            ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)),
            'reflect',
        )
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(
            1.0 - np.minimum(np.float32(x) / pad[0],
            np.float32(w - 1 - x) / pad[2]),
            1.0 - np.minimum(np.float32(y) / pad[1],
            np.float32(h - 1 - y) / pad[3]),
        )
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(
            img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # transform
    quad = (quad + 0.5).flatten()
    img = img.transform(
        (transform_size, transform_size),
        Image.QUAD,
        quad,
        Image.BILINEAR,
    )
    if output_size < transform_size:
        img = img.resize((output_size, output_size), Image.ANTIALIAS)

    return img


def unalign_and_uncrop(img, wrapper_aligned, cna, transform_size):
    cna = cna.resize(
        (transform_size, transform_size),
        Image.ANTIALIAS,
    )
    cna_padded, offset = _pad(cna, wrapper_aligned)
    wrapper_aligned_padded = _shift_coords(wrapper_aligned, offset)
    cna_mask = _create_rect_mask(cna_padded.size, offset, transform_size)
    cropped = unalign(cna_padded, wrapper_aligned_padded, img.size)
    cropped_mask = unalign(cna_mask, wrapper_aligned_padded, img.size)
    return uncrop(cropped, cropped_mask, img)


def unalign(aligned, quad, output_size):
    return aligned.transform(output_size, Image.QUAD, quad, Image.BILINEAR)


def uncrop(cropped, cropped_mask, full_img):
    return Image.composite(cropped, full_img, cropped_mask)


def align_coords(coords, quad, transform_size):
    nw = align_point(coords[0], coords[1], quad, transform_size)
    sw = align_point(coords[2], coords[3], quad, transform_size)
    se = align_point(coords[4], coords[5], quad, transform_size)
    ne = align_point(coords[6], coords[7], quad, transform_size)
    return nw + sw + se + ne


def unalign_coords(coords, quad, transform_size):
    nw = unalign_point(coords[0], coords[1], quad, transform_size)
    sw = unalign_point(coords[2], coords[3], quad, transform_size)
    se = unalign_point(coords[4], coords[5], quad, transform_size)
    ne = unalign_point(coords[6], coords[7], quad, transform_size)
    return nw + sw + se + ne


def align_point(x_prime, y_prime, quad, transform_size):
    w = transform_size
    h = transform_size

    # name of each coorinate in quad
    # (e.g. x coord of southwest corner = swx)
    nwx, nwy, swx, swy, sex, sey, nex, ney = quad

    # deltas between the coords
    # (e.g. change in x coord between the two north corners = dnx)
    dnx = nex - nwx
    dwx = swx - nwx
    dsx = sex - swx
    dny = ney - nwy
    dwy = swy - nwy
    dsy = sey - swy

    # define some shorthands that will be useful
    a0 = nwx
    a1 = dnx / w
    a2 = dwx / h
    a3 = (dsx - dnx) / (w * h)
    b0 = nwy
    b1 = dny / w
    b2 = dwy / h
    b3 = (dsy - dny) / (w * h)

    '''
    the problem can now be defined as:

    solve the following system of equations for x and y

    x_prime = a0 + a1 * x + a2 * y + a3 * x * y
    y_prime = b0 + b1 * x + b2 * y + b3 * x * y
    '''

    # additional shorthands for the next step
    p = b2 * a3 - b3 * a2
    q = (b0 * a3 - b3 * a0) + (b2 * a1 - b1 * a2) + (b3 * x_prime - a3 * y_prime)
    r = (b0 * a1 - b1 * a0) + (b1 * x_prime - a1 * y_prime)

    '''
    solving for y without any x terms yields this quadradic equation:

    p * y^2 + q * y + r = 0

    if p == 0, the solution is:

    y = -r / q

    otherwise, the solution is:

    y = (-q +/- sqrt(q^2 - 4 * p * r)) / (2 * p)

    after inspection, it seems +/- should always be treated as +
    (likely because of the constraints imposed via the ordering of the corners)
    '''
    if p == 0.:
        y = -r / q
    else:
        sqrt_term = math.sqrt(q ** 2 - 4 * p * r)
        y = (-q + sqrt_term) / (2 * p)
        # y = (-q - sqrt_term) / (2 * p)

    # using this solution for y, solving for x yields:
    x = (x_prime - a0 - a2 * y) / (a1 + a3 * y)

    if USE_PIXEL_CENTER:
        x -= 0.5
        y -= 0.5

    return x, y


def unalign_point(x, y, quad, transform_size):
    nwx, nwy, swx, swy, sex, sey, nex, ney = quad
    w = transform_size
    h = transform_size

    if USE_PIXEL_CENTER:
        x += 0.5
        y += 0.5

    x_prime = nwx + x * (nex - nwx) / w + y * (swx - nwx) / h + \
              x * y * (sex - swx - nex + nwx) / (w * h)

    y_prime = nwy + x * (ney - nwy) / w + y * (swy - nwy) / h + \
              x * y * (sey - swy - ney + nwy) / (w * h)

    return x_prime, y_prime


def _shift_coords(coords, offset):
    dx, dy = offset
    shifted_coords = []
    for i, val in enumerate(coords):
        if i % 2:
            shifted = coords[i] + dy
        else:
            shifted = coords[i] + dx
        shifted_coords.append(shifted)
    return shifted_coords


def _create_rect_mask(full_size, offset, transform_size):
    mask = Image.new('1', full_size, 0)
    draw = ImageDraw.Draw(mask)
    offset_x, offset_y = offset
    draw.rectangle((
        offset_x + ALIGNED_MASK_BUFFER,
        offset_y + ALIGNED_MASK_BUFFER,
        offset_x + transform_size - ALIGNED_MASK_BUFFER,
        offset_y + transform_size - ALIGNED_MASK_BUFFER,
    ), fill=1)
    return mask


def _pad(img, quad):
    offset_x = -min(quad[::2])
    offset_y = -min(quad[1::2])
    offset = (offset_x, offset_y)
    padded_w = max(quad[::2]) + offset_x
    padded_h = max(quad[1::2]) + offset_y
    padded_img = Image.new(img.mode, (padded_w, padded_h), (255, 255, 255))
    padded_img.paste(img, offset)
    return padded_img, offset
