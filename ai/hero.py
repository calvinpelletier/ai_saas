import os
from PIL import Image
import numpy as np
from pathlib import Path
import fire
import json
import torch

from ai_saas.constants import DATA_PATH
from ai_saas.worker.align import Align
from ai_saas.worker.detect import Detect
from ai_saas.worker.encode import Encode
from ai_saas.worker.finetune import Finetune
from ai_saas.worker.process import Process
from ai_saas.worker.inpaint2 import Inpaint
from ai_saas.worker.unalign import Unalign
from ai_saas.ai.util.etc import pil_to_tensor, create_img_row
from ai_saas.ai.util.pretrained import build_pretrained_sg2


DIR = Path(DATA_PATH) / 'hero'
OG_FNAME = 'og.jpg'
FACE_IDX = 2
DELTA_G_SIGN = -1
N_PREVIEWS = 8
DELTA_G_MAG = 0.7
N_FRAMES = 24


class Hero:
    def pre(self):
        og = Image.open(DIR / OG_FNAME)

        detect = Detect()
        with detect.context():
            faces = detect(og)[0]
        face = faces[FACE_IDX]

        align = Align()
        with align.context():
            cna, outer_cna, ucna_coords = align(og, face)
        cna.save(DIR / 'cna.png')
        outer_cna.save(DIR / 'outer_cna.png')
        with open(DIR / 'ucna_coords.json', 'w') as f:
            json.dump(ucna_coords, f)

        encode = Encode()
        cna_tensor = pil_to_tensor(cna)
        with encode.context():
            w_plus = encode(cna_tensor)[0]
        with open(DIR / 'w_plus.json', 'w') as f:
            json.dump(w_plus.cpu().numpy().tolist(), f)

        finetune = Finetune()
        with finetune.context():
            ft_G, ft_encs = finetune(cna_tensor, w_plus, DELTA_G_SIGN)
        with open(DIR / 'ft_encs.json', 'w') as f:
            json.dump({
                k: v.squeeze(0).detach().cpu().numpy().tolist()
                for k, v in ft_encs.items()
            }, f)
        torch.save(ft_G.state_dict(), DIR / 'ft_g.pt')

        process = Process()
        with process.context():
            print(list(np.linspace(0, 1, N_PREVIEWS)))
            previews = [
                process(ft_encs, delta_g_mag)[0]
                for delta_g_mag in np.linspace(0, 1, N_PREVIEWS)
            ]
            create_img_row(previews, 256).save(DIR / 'previews.png')

    def post(self):
        og, outer_cna, ucna_coords, w_plus, ft_G, overrides = load()

        inpaint = Inpaint()
        with inpaint.context():
            new_cna, inpainted = inpaint(
                outer_cna,
                w_plus,
                ft_G,
                DELTA_G_SIGN,
                DELTA_G_MAG,
                overrides,
            )
        new_cna.save(DIR / 'new_cna.png')
        inpainted.save(DIR / 'inpainted.png')

        unalign = Unalign()
        with unalign.context():
            final, final_wm = unalign(og, inpainted, ucna_coords)
        final.save(DIR / 'final.png')
        final_wm.save(DIR / 'final_wm.png')

        w, h = og.size
        comparison = Image.new('RGB', (w * 2, h), 'black')
        comparison.paste(og, (0, 0))
        comparison.paste(final, (w, 0))
        comparison.save(DIR / 'comparison.png')

    def frames(self):
        og, outer_cna, ucna_coords, w_plus, ft_G, overrides = load()

        inpaint = Inpaint()
        unalign = Unalign()

        dir = DIR / 'frames'
        dir.mkdir(exist_ok=True)
        for i, x in enumerate(np.linspace(0, 1, N_FRAMES)):
            x = 1 / (1 + np.exp(-16*x + 8))

            with inpaint.context():
                new_cna, inpainted = inpaint(
                    outer_cna,
                    w_plus,
                    ft_G,
                    DELTA_G_SIGN,
                    x * DELTA_G_MAG,
                    overrides,
                )

            with unalign.context():
                final, final_wm = unalign(og, inpainted, ucna_coords)
            final.save(dir / f'{i}.png')


def load():
    og = Image.open(DIR / OG_FNAME)

    outer_cna = Image.open(DIR  / 'outer_cna.png')

    with open(DIR / 'ucna_coords.json', 'r') as f:
        ucna_coords = json.load(f)

    with open(DIR / 'w_plus.json', 'r') as f:
        w_plus = torch.from_numpy(np.asarray(
            json.load(f)
        )).to(torch.float32).to('cuda')

    ft_G = build_pretrained_sg2().synthesis
    ft_G.load_state_dict(torch.load(DIR / 'ft_g.pt'))

    overrides = [
        load_mask('gt_mask_override.png'),
        load_mask('inpaint_mask_override.png'),
    ]

    return og, outer_cna, ucna_coords, w_plus, ft_G, overrides


def load_mask(fname):
    mask = Image.open(
        DIR / fname,
    ).resize((512, 512), Image.LANCZOS)
    mask = torch.from_numpy(np.copy(np.asarray(
        mask,
    ).transpose(2, 0, 1))).to('cuda').to(torch.float32) / 255.
    mask = (mask[0, :, :] > 0.5).float()
    return mask


if __name__ == '__main__':
    fire.Fire(Hero)
