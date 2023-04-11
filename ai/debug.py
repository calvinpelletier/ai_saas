import os
from PIL import Image, ImageDraw

from asi.util.etc import pil_to_tensor, tensor_to_pil
from asi.unit.seg.seg import colorize_seg, binary_seg_to_img

from ai_saas.lib.common.util import get_environ


DIR = os.path.join(get_environ('ASI_DATA'), 'debug')


def encode(w):
    _export_tensor_glimpse(w, 'w')

    g = build_pretrained_sg2().synthesis
    img = g(w, noise_mode='const')
    img = tensor_to_pil(img[0])
    img.save(_path('cna_rec.png'))


def finetune(w, g):
    img = g(w, noise_mode='const')
    img = tensor_to_pil(img[0])
    img.save(_path('cna_rec_ft.png'))


def process(delta_g, new_w, new_cna):
    _export_val(delta_g, 'delta_g')
    _export_tensor_glimpse(new_w, 'new_w.txt')
    tensor_to_pil(new_cna[0]).save(_path('new_cna.png'))


def segs(outer_cna, new_cna, outer_fhbc, new_fhbc, new_outer_fhbc):
    _export_seg(tensor_to_pil(outer_cna[0]), outer_fhbc, 'outer_seg')
    _export_seg(tensor_to_pil(new_cna[0]), new_fhbc, 'inner_seg')
    _export_seg(None, new_outer_fhbc, 'outer_seg_pred')


def masks(img, inner_gan_mask, gt_mask, inpaint_mask):
    img = tensor_to_pil(img[0])
    _export_mask(img, inner_gan_mask, 'inner_gan')
    _export_mask(img, gt_mask, 'gt')
    _export_mask(img, inpaint_mask, 'inpaint')


def inpaint(pre_inpaint, inpainted):
    tensor_to_pil(pre_inpaint[0]).save(_path('pre_inpaint.png'))
    tensor_to_pil(inpainted[0]).save(_path('inpainted.png'))


def final(final):
    final.save(_path('final.png'))


def draw_quad(draw, quad, color, lw):
    draw.line((quad[0], quad[1], quad[2], quad[3]), fill=color, width=lw)
    draw.line((quad[2], quad[3], quad[4], quad[5]), fill=color, width=lw)
    draw.line((quad[4], quad[5], quad[6], quad[7]), fill=color, width=lw)
    draw.line((quad[6], quad[7], quad[0], quad[1]), fill=color, width=lw)


def export_val(val, name):
    with open(_path(f'{name}.txt'), 'w') as f:
        f.write(str(val) + '\n')


def export_tensor_glimpse(tensor, name):
    with open(path(f'{name}.txt'), 'w') as f:
        f.write(str(tensor.cpu().numpy()) + '\n')
        f.write(str(tensor.shape) + '\n')


def export_seg(img, seg, name):
    h, w = seg.shape
    seg = seg.unsqueeze(0)
    if img is None:
        colorize_seg(seg, needs_argmax=False)[0].save(path(f'{name}.png'))
    else:
        img = img.copy().convert('RGBA').resize((w, h))
        seg_img = colorize_seg(seg, needs_argmax=False)[0].convert('RGBA')
        seg_img.putalpha(127)
        Image.alpha_composite(img, seg_img).save(path(f'{name}.png'))


def export_mask(img, mask, name):
    h, w = mask.shape
    img = img.copy().convert('RGBA').resize((w, h))
    # binary_seg_to_img(mask).save(_path(f'{name}_mask.png'))
    mask_img = binary_seg_to_img(mask).convert('RGBA')
    mask_img.putalpha(127)
    Image.alpha_composite(img, mask_img).save(path(f'{name}_mask.png'))


def path(fname):
    return os.path.join(DIR, fname)























#
