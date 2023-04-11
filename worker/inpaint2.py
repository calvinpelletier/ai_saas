import torch
import numpy as np
import sys
from pathlib import Path

from ai_saas.lib.worker import Worker
from ai_saas.server.db.client import DatabaseClient
from ai_saas.ai.util.pretrained import build_pretrained_sg2
from ai_saas.ai.util.etc import tensor_to_pil, resize, pil_to_tensor
from ai_saas.ai.constants import MAX_MAG, INPAINT_SIZE, INNER_SEG_PRED_IMSIZE, \
    INNER_IMSIZE, SEG_PRED_IMSIZE
from ai_saas.ai.util.seg import fhbc_seg_to_fh_mask
from ai_saas.ai.util.mask import union, intersection, dilate, inverse, \
    get_inner_mask, get_outer_boundary_mask, erode
from ai_saas.ai.util.outer import get_dilate_kernel
from ai_saas.lib.etc import AttrDict
from ai_saas.ai.model.inpaint.aot import AotInpainter
from ai_saas.ai.model.seg.seg import FHBC_GROUPS, Segmenter
from ai_saas.ai.model.seg.outer_seg import OuterSegPredictor
from ai_saas.lib.dep import download_model
from ai_saas.lib.ai import build_f_w
from ai_saas.constants import DATA_PATH
from ai_saas.ai.diffusion_inpaint import DiffusionInpainter


INNER_MASK = get_inner_mask(INNER_IMSIZE, INPAINT_SIZE)
INV_OUTER_BOUNDARY_MASK = 1. - get_outer_boundary_mask(INPAINT_SIZE).to('cuda')
DILATE_KERNEL = get_dilate_kernel(INPAINT_SIZE)


class InpaintWorker(Worker):
    def __init__(self, env):
        super().__init__('inpaint', env, Inpaint(), needs_db=True)
        self.device = self.cfg.task.device


    def pre(self, iid):
        w = torch.from_numpy(np.asarray(
            self.storage.read_from_img_dir(iid, 'w_plus.json'),
        )).to(torch.float32).to(self.device)

        img_data = self.db.get_image(iid)['image']
        delta_g_sign = img_data['si']['delta_g_sign']
        print(img_data['di']['delta_g_mag'])
        delta_g_mag = float(img_data['di']['delta_g_mag'])

        G = build_pretrained_sg2().synthesis
        G.load_state_dict(torch.load(
            self.storage.read_from_img_dir(iid, 'ft_g.pt'),
        ))

        outer_cna = self.storage.read_from_img_dir(iid, 'outer_cna.png')

        return outer_cna, w, G, delta_g_sign, delta_g_mag


    def post(self, iid, new_cna, inpainted):
        self.storage.write_to_img_dir(iid, 'new_cna.png', new_cna)
        self.storage.write_to_img_dir(iid, 'inpainted.png', inpainted)


class Inpaint:
    def __init__(self, device='cuda'):
        self.device = device

        self.F_w = build_f_w(self.device)

        self.osp = OuterSegPredictor(
            AttrDict({'dataset': {
                'imsize': 192,
                'inner_imsize': 128,
                'n_labels': 4,
            }}),
            pred_from_seg_only=False,
            nc_base=32,
        )
        self.osp.load_state_dict(
            torch.load(download_model('fp/outer_seg.pt'))['model'],
        )
        self.osp = self.osp.to(self.device).eval()

        self.segmenter = Segmenter().to(self.device).eval()

        # self.inpainter = AotInpainter(
        #     AttrDict({'dataset': AttrDict({'imsize': 512})}),
        # )
        # self.inpainter.load_state_dict(torch.load(download_model('aot/G.pt')))
        # self.inpainter = self.inpainter.to(self.device).eval()

        self.inpainter = DiffusionInpainter(
            Path(DATA_PATH) / 'models' / 'palette' / '200_Network.pth',
        )


    def context(self):
        return torch.no_grad()


    def __call__(self, outer_cna, w, G, delta_g_sign, delta_g_mag, overrides):
        outer_cna = pil_to_tensor(outer_cna)

        delta_g_sign = torch.tensor(
            delta_g_sign,
        ).to(torch.float32).to(self.device)
        delta_g_mag = torch.tensor(
            delta_g_mag * MAX_MAG,
        ).to(torch.float32).to(self.device)
        delta_g = delta_g_sign * delta_g_mag

        new_cna = G(self.F_w(w.unsqueeze(0), delta_g), noise_mode='const')

        outer_fhbc = self.seg(outer_cna, INPAINT_SIZE)
        new_inner_fhbc = self.seg(new_cna, INNER_SEG_PRED_IMSIZE)
        new_outer_fhbc = self.predict_outer_seg(new_cna, new_inner_fhbc)

        gt_override, inpaint_override = overrides
        gen_mask, inpaint_mask = calc_masks(
            outer_fhbc,
            new_outer_fhbc,
            gt_override,
            inpaint_override,
        )
        new_cna_outer = inner_to_outer(
            new_cna,
            INNER_MASK,
            INNER_IMSIZE,
            INPAINT_SIZE,
        )
        pre_inpaint = new_cna_outer * gen_mask + \
            resize(outer_cna, INPAINT_SIZE) * inverse(gen_mask)

        inpainted = self.inpainter(
            tensor_to_pil(pre_inpaint),
            inpaint_mask.unsqueeze(0),
        )
        inpainted = inpainted * inpaint_mask + \
            pre_inpaint * inverse(inpaint_mask)

        # pre_inpaint = pre_inpaint * inverse(inpaint_mask)
        return tensor_to_pil(new_cna), tensor_to_pil(inpainted)


    def seg(self, img, output_imsize):
        return torch.argmax(self.segmenter(
            img,
            groups=FHBC_GROUPS,
            output_imsize=output_imsize,
        )[0], dim=0)


    def predict_outer_seg(self, inner_img, inner_seg):
        outer_seg = self.osp(
            inner_to_outer(
                inner_seg.unsqueeze(0),
                self.osp.inner_mask,
                INNER_SEG_PRED_IMSIZE,
                SEG_PRED_IMSIZE,
                is_seg=True,
            ),
            inner_to_outer(
                inner_img,
                self.osp.inner_mask,
                INNER_SEG_PRED_IMSIZE,
                SEG_PRED_IMSIZE,
            ),
        )
        return torch.argmax(
            resize(outer_seg, INPAINT_SIZE)[0],
            dim=0,
        )


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


def calc_masks(outer_fhbc, new_outer_fhbc, gt_override, inpaint_override):
    # fh = fhbc_seg_to_fh_mask(outer_fhbc)
    new_fh = fhbc_seg_to_fh_mask(new_outer_fhbc)
    # fh_union = union(fh, new_fh)

    # dilated_fh = dilate(fh, DILATE_KERNEL)
    # new_dilated_fh = dilate(new_fh, DILATE_KERNEL)
    # dilated_fh_union = union(dilated_fh, new_dilated_fh)

    gen_mask = intersection(new_fh, INNER_MASK)

    inpaint_mask = union(intersection(
        dilate(gen_mask, DILATE_KERNEL),
        inverse(erode(gen_mask, DILATE_KERNEL)),
    ), inpaint_override)

    # inner_gan_mask = intersection(new_fh, INNER_MASK)
    # inv_inner_gan_mask = inverse(inner_gan_mask)
    #
    # inpaint_mask = intersection(
    #     dilated_fh_union,
    #     inv_inner_gan_mask,
    #     INV_OUTER_BOUNDARY_MASK,
    #     inv_gt_override,
    # )
    #
    # gt_mask = intersection(
    #     inverse(inpaint_mask),
    #     inv_inner_gan_mask,
    # )
    return gen_mask, inpaint_mask


if __name__ == '__main__':
    worker = InpaintWorker(sys.argv[1])
    worker.start()
