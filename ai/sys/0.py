import torch
from PIL import Image, ImageDraw
from time import time
from functools import wraps
import os

from asi.trainer.pti import PtiTrainer
from asi.util.outer import get_inner_mask, get_dilate_kernel, \
    get_outer_boundary_mask
from asi.util.etc import pil_to_tensor, normalized_tensor_to_pil_img
from asi.unit.seg.seg import FHBC_GROUPS, colorize_seg, binary_seg_to_img
from asi.util.pretrained import build_pretrained_sg2

from ai_saas.lib.etc import AttrDict
from ai_saas.lib.etc.util import weighted_add
from ai_saas.lib.etc.cache import lru_cache
from ai_saas.lib.ai.util.img import resize
from ai_saas.lib.ai.util.mask import union, intersection, dilate, inverse

from ai_saas.ai.face import predict_landmarks, landmarks_to_alignment_coords
from ai_saas.ai.cna import crop_and_align, unalign_and_uncrop, unalign_coords, \
    align_coords
from ai_saas.ai.seg.util import fhbc_seg_to_fh_mask


# TODO
def inner_to_outer(
    img,
    inner_mask,
    inner_imsize,
    outer_imsize,
    is_seg=False,
):
    if is_seg:
        assert img.shape == (1, inner_imsize, inner_imsize)
        ret = torch.zeros(1, outer_imsize, outer_imsize, dtype=torch.long,
            device='cuda')
    else:
        img = resize(img, inner_imsize)
        ret = torch.zeros(1, 3, outer_imsize, outer_imsize, device='cuda')

    buf = (outer_imsize - inner_imsize) // 2
    for y in range(outer_imsize):
        for x in range(outer_imsize):
            is_inner = y >= buf and y < (buf + inner_imsize) and \
                x >= buf and x < (buf + inner_imsize)
            if is_inner:
                assert inner_mask[y][x] == 1.
                if is_seg:
                    ret[0, y, x] = img[0, y - buf, x - buf]
                else:
                    ret[0, :, y, x] = img[0, :, y - buf, x - buf]
            else:
                assert inner_mask[y][x] == 0.
    return ret


# TODO
def calc_outer_coords_aligned(align_transform_size):
    buf = align_transform_size // 4
    return [
        -buf, -buf, # nw
        -buf, align_transform_size + buf, # sw
        align_transform_size + buf, align_transform_size + buf, # se
        align_transform_size + buf, -buf # ne
    ]


# TODO
def gate(f):
    @wraps(f)
    def wrapper(self, *args, **kwargs):
        if not self._enabled:
            return
        return f(self, *args, **kwargs)
    return wrapper

class Debug:
    def __init__(self, dir):
        self._dir = dir
        self._enabled = False
        self._sg2 = None
        self._count = None


    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, val):
        self._enabled = val
        if self._enabled:
            if self._sg2 is None:
                self._sg2 = build_pretrained_sg2().synthesis
        else:
            self._sg2 = None

    def path(self, fname, is_first=False):
        if is_first:
            self._count = 0
        else:
            self._count += 1
        return os.path.join(self._dir, f'{self._count:03d}{fname}')


    @gate
    def pre_cna(self, img, box, landmarks, quad):
        img = img.copy()
        w, h = img.size
        draw = ImageDraw.Draw(img)

        draw.rectangle((
            int(box['x'] * w),
            int(box['y'] * h),
            int((box['x'] + box['w']) * w),
            int((box['y'] + box['h']) * h),
        ), outline='green', width=int(box['w'] * w * 0.02))

        rad = int(box['w'] * w * 0.01)
        for x, y in landmarks:
            draw.ellipse((
                x - rad,
                y - rad,
                x + rad,
                y + rad
            ), fill='blue', outline='blue')

        self._draw_quad(draw, quad, 'red', int(box['w'] * w * 0.02))

        img.save(self.path('pre_cna.png', is_first=True))


    @gate
    def cna(self, img):
        img.save(self.path('cna.png'))


    @gate
    def prep_inner(self, w, g):
        self._export_tensor_glimpse(w, 'w')

        img = self._sg2(w, noise_mode='const')
        img = normalized_tensor_to_pil_img(img[0])
        img.save(self.path('cna_rec.png'))

        img = g(w, noise_mode='const')
        img = normalized_tensor_to_pil_img(img[0])
        img.save(self.path('cna_rec_ft.png'))


    @gate
    def prep_outer(self, img, quad, cna, wrapper_aligned):
        img = img.copy()
        draw = ImageDraw.Draw(img)
        lw = int((quad[2] - quad[0]) * 0.5)
        self._draw_quad(draw, quad, 'red', lw)
        self._draw_quad(draw, wrapper_aligned, 'blue', lw)
        img.save(self.path('prep_outer.png'))

        normalized_tensor_to_pil_img(cna[0]).save(self.path('outer_cna.png'))


    @gate
    def process(self, delta_g, new_w, new_cna):
        self._export_val(delta_g, 'delta_g')
        self._export_tensor_glimpse(new_w, 'new_w.txt')
        normalized_tensor_to_pil_img(new_cna[0]).save(self.path('new_cna.png'))


    @gate
    def segs(self, outer_cna, new_cna, outer_fhbc, new_fhbc, new_outer_fhbc):
        self._export_seg(
            normalized_tensor_to_pil_img(outer_cna[0]), outer_fhbc, 'outer_seg')
        self._export_seg(
            normalized_tensor_to_pil_img(new_cna[0]), new_fhbc, 'inner_seg')
        self._export_seg(None, new_outer_fhbc, 'outer_seg_pred')


    @gate
    def masks(self, img, inner_gan_mask, gt_mask, inpaint_mask):
        img = normalized_tensor_to_pil_img(img[0])
        self._export_mask(img, inner_gan_mask, 'inner_gan')
        self._export_mask(img, gt_mask, 'gt')
        self._export_mask(img, inpaint_mask, 'inpaint')


    @gate
    def inpaint(self, pre_inpaint, inpainted):
        normalized_tensor_to_pil_img(pre_inpaint[0]).save(
            self.path('pre_inpaint.png'))
        normalized_tensor_to_pil_img(inpainted[0]).save(
            self.path('inpainted.png'))


    @gate
    def final(self, final):
        final.save(self.path('final.png'))


    def _draw_quad(self, draw, quad, color, lw):
        draw.line((quad[0], quad[1], quad[2], quad[3]), fill=color, width=lw)
        draw.line((quad[2], quad[3], quad[4], quad[5]), fill=color, width=lw)
        draw.line((quad[4], quad[5], quad[6], quad[7]), fill=color, width=lw)
        draw.line((quad[6], quad[7], quad[0], quad[1]), fill=color, width=lw)


    def _export_val(self, val, name):
        with open(self.path(f'{name}.txt'), 'w') as f:
            f.write(str(val) + '\n')


    def _export_tensor_glimpse(self, tensor, name):
        with open(self.path(f'{name}.txt'), 'w') as f:
            f.write(str(tensor.cpu().numpy()) + '\n')
            f.write(str(tensor.shape) + '\n')


    def _export_seg(self, img, seg, name):
        h, w = seg.shape
        seg = seg.unsqueeze(0)
        if img is None:
            colorize_seg(seg, needs_argmax=False)[0].save(
                self.path(f'{name}.png'))
        else:
            img = img.copy().convert('RGBA').resize((w, h))
            seg_img = colorize_seg(seg, needs_argmax=False)[0].convert('RGBA')
            seg_img.putalpha(127)
            Image.alpha_composite(img, seg_img).save(self.path(f'{name}.png'))


    def _export_mask(self, img, mask, name):
        h, w = mask.shape
        img = img.copy().convert('RGBA').resize((w, h))
        binary_seg_to_img(mask).save(self.path(f'{name}_mask.png'))
        # mask_img = binary_seg_to_img(mask).convert('RGBA')
        # mask_img.putalpha(127)
        # Image.alpha_composite(img, mask_img).save(self.path(f'{name}_mask.png'))


class AiSystem:
    def __init__(self, cfg, model_manager):
        self.cfg = cfg
        self.mm = model_manager

    def get_model(self, key):
        return self.mm.get(getattr(self.cfg.models, key))

    def infer(self, key, *args, **kwargs):
        with torch.no_grad():
            ret = self.get_model(key)(*args, **kwargs)
        return ret


class Sys0(AiSystem):
    def __init__(self, cfg, model_manager):
        super().__init__(cfg, model_manager)

        # TODO
        self.pti_trainer = PtiTrainer('cuda')

        # TODO
        self.INNER_IMSIZE = self.outer_to_inner_imsize(self.cfg.inpaint_size)
        self.INNER_MASK = get_inner_mask(
            self.INNER_IMSIZE, self.cfg.inpaint_size)
        self.OUTER_COORDS_ALIGNED = calc_outer_coords_aligned(
            self.cfg.align_transform_size)
        self.OUTER_CNA_SIZE = int(
            self.cfg.outer_inner_ratio * self.cfg.cna_size)
        self.DILATE_KERNEL = get_dilate_kernel(self.cfg.inpaint_size)
        self.INV_OUTER_BOUNDARY_MASK = 1. - get_outer_boundary_mask(
            self.cfg.inpaint_size).to('cuda')

        self.debug = Debug('/home/asiu/data/debug')


    def __call__(self, inf_type, si, di):
        self.debug.enabled = inf_type == 'debug'
        prep = self.preprocess(si.id, si)
        preview = self.process(si, di, prep)
        if inf_type == 'preview':
            return preview
        final = self.postprocess(si, di, prep, preview)
        return final


    @lru_cache(maxsize=1024)
    def preprocess(self, si):
        landmarks = predict_landmarks(
            self.get_model('face_landmark_predictor'),
            si.img,
            si.face_box,
        )
        inner_coords = landmarks_to_alignment_coords(landmarks)
        self.debug.pre_cna(si.img, si.face_box, landmarks, inner_coords)
        cna = crop_and_align(
            si.img,
            inner_coords,
            self.cfg.cna_size,
            self.cfg.align_transform_size,
        )
        self.debug.cna(cna)
        cna_tensor = pil_to_tensor(cna)
        return AttrDict({
            **self.prep_inner(cna_tensor),
            **self.prep_outer(si.img, inner_coords),
        })


    def process(self, si, di, prep):
        delta_g = si.delta_g_sign * di.delta_g_mag
        new_w = self.infer('f', prep.w, delta_g)

        # TEMP
        from asi.util.factory import build_model_from_exp
        from asi.unit.lerp.onnx import OnnxLevelsDynamicLerper
        f_model, f_cfg = build_model_from_exp('lerp/5/5', 'G')
        f2 = OnnxLevelsDynamicLerper(
            levels=f_cfg.model.G.levels,
            final_activation=f_cfg.model.G.final_activation,
            mult=f_cfg.model.G.mult,
            lr_mul=f_cfg.model.G.lr_mul,
            pixel_norm=f_cfg.model.G.pixel_norm,
        )
        f2.load_state_dict(f_model.f.state_dict())
        f2 = f2.to('cuda').eval()
        new_w2 = f2(prep.w, torch.tensor(1.).to('cuda').unsqueeze(0))
        assert (new_w2 == new_w).all()

        G = prep.G.to('cuda')
        new_cna = G(new_w)
        self.debug.process(delta_g, new_w, new_cna)
        return new_cna


    def postprocess(self, si, di, prep, preview):
        outer_fhbc = self.seg(prep.outer_cna, self.cfg.inpaint_size)
        new_inner_fhbc = self.seg(
            preview,
            self.outer_to_inner_imsize(self.cfg.seg_pred_imsize),
        )
        new_outer_fhbc = self.predict_outer_seg(preview, new_inner_fhbc)
        self.debug.segs(
            prep.outer_cna, preview, outer_fhbc, new_inner_fhbc, new_outer_fhbc)

        inner_gan_mask, gt_mask, inpaint_mask = self.calc_masks(
            outer_fhbc, new_outer_fhbc)
        self.debug.masks(prep.outer_cna, inner_gan_mask, gt_mask, inpaint_mask)

        # pre_inpaint = weighted_add(
        #     (resize(prep.outer_cna, self.cfg.inpaint_size), gt_mask),
        #     (inner_to_outer(
        #         preview,
        #         self.INNER_MASK,
        #         self.INNER_IMSIZE,
        #         self.cfg.inpaint_size,
        #     ), inner_gan_mask),
        # )
        pre_inpaint = resize(prep.outer_cna, self.cfg.inpaint_size) * gt_mask \
            + inner_to_outer(
                preview,
                self.INNER_MASK,
                self.INNER_IMSIZE,
                self.cfg.inpaint_size,
            ) * inner_gan_mask
        pre_inpaint *= inverse(inpaint_mask)
        print('pre_inpaint', pre_inpaint, pre_inpaint.shape)
        inpainted = self.infer(
            'inpainter',
            pre_inpaint,
            inpaint_mask.unsqueeze(0),
        )
        self.debug.inpaint(pre_inpaint, inpainted)

        final = unalign_and_uncrop(
            si.img,
            prep.wrapper_coords_aligned,
            normalized_tensor_to_pil_img(inpainted[0]),
            self.cfg.align_transform_size,
        )
        self.debug.final(final)
        return final


    def prep_inner(self, cna):
        cna_256 = resize(cna, 256)
        w = self.infer('e', cna_256)
        G = self.pti_trainer.train(cna.to('cuda'), w.to('cuda')).synthesis
        self.debug.prep_inner(w, G)
        return {
            'w': w,
            'G': G.to('cpu'),
        }


    def prep_outer(self, img, inner_coords):
        outer_coords = unalign_coords(
            self.OUTER_COORDS_ALIGNED,
            inner_coords,
            self.cfg.align_transform_size,
        )
        outer_cna = pil_to_tensor(crop_and_align(
            img,
            outer_coords,
            self.OUTER_CNA_SIZE,
            self.cfg.align_transform_size,
        ))
        w, h = img.size
        wrapper_coords_aligned = align_coords(
            (
                0, 0,  # nw
                0, h,  # sw
                w, h,  # se
                w, 0,  # ne
            ),
            outer_coords,
            self.cfg.align_transform_size,
        )
        wrapper_coords_aligned = [
            int(round(x)) for x in list(wrapper_coords_aligned)
        ]
        self.debug.prep_outer(
            img, outer_coords, outer_cna, wrapper_coords_aligned)
        return {
            'outer_cna': outer_cna,
            'wrapper_coords_aligned': wrapper_coords_aligned,
        }


    def seg(self, img, output_imsize):
        return torch.argmax(self.infer('segmenter',
            img,
            groups=FHBC_GROUPS,
            output_imsize=output_imsize,
        )[0], dim=0)


    def predict_outer_seg(self, inner_img, inner_seg):
        outer_seg_predictor = self.get_model('outer_seg_predictor')
        inner_size = self.outer_to_inner_imsize(self.cfg.seg_pred_imsize)
        outer_seg = outer_seg_predictor(
            inner_to_outer(
                inner_seg.unsqueeze(0),
                outer_seg_predictor.inner_mask,
                inner_size,
                self.cfg.seg_pred_imsize,
                is_seg=True,
            ),
            inner_to_outer(
                inner_img,
                outer_seg_predictor.inner_mask,
                inner_size,
                self.cfg.seg_pred_imsize,
            ),
        )
        return torch.argmax(
            resize(outer_seg, self.cfg.inpaint_size)[0],
            dim=0,
        )


    def calc_masks(self, outer_fhbc, new_outer_fhbc):
        fh = fhbc_seg_to_fh_mask(outer_fhbc)
        new_fh = fhbc_seg_to_fh_mask(new_outer_fhbc)
        inner_gan_mask = intersection(new_fh, self.INNER_MASK)
        inv_inner_gan_mask = inverse(inner_gan_mask)
        dilated_fh = dilate(fh, self.DILATE_KERNEL)
        new_dilated_fh = dilate(new_fh, self.DILATE_KERNEL)
        dilated_fh_union = union(dilated_fh, new_dilated_fh)
        inpaint_mask = intersection(
            dilated_fh_union,
            inv_inner_gan_mask,
            self.INV_OUTER_BOUNDARY_MASK,
        )
        gt_mask = intersection(inverse(inpaint_mask), inv_inner_gan_mask)
        return inner_gan_mask, gt_mask, inpaint_mask


    def outer_to_inner_imsize(self, outer_imsize):
        return int(outer_imsize / self.cfg.outer_inner_ratio)









































#
