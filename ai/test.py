import os
from PIL import Image
import numpy as np

from ai_saas.constants import DATA_PATH
from ai_saas.worker.align import Align
from ai_saas.worker.detect import Detect
from ai_saas.worker.encode import Encode
from ai_saas.worker.finetune import Finetune
from ai_saas.worker.process import Process
from ai_saas.worker.inpaint import Inpaint
from ai_saas.worker.unalign import Unalign
from ai_saas.ai.util.etc import pil_to_tensor, create_img_row


TEST_DIR = os.path.join(DATA_PATH, 'imgs/test')
DELTA_G_SIGN = -1
N_PREVIEWS = 8


class TestAisys:
    def __init__(self):
        self.detect = Detect()
        self.align = Align()
        self.encode = Encode()
        self.finetune = Finetune()
        self.process = Process()
        self.inpaint = Inpaint()
        self.unalign = Unalign()

    def __call__(self, dir, og_fname, face_idx=None, delta_g_mag=None):
        og = Image.open(os.path.join(dir, og_fname))

        if face_idx is None:
            with open(os.path.join(dir, 'face_idx.txt'), 'r') as f:
                face_idx = int(f.read().strip())

        if delta_g_mag is None:
            with open(os.path.join(dir, 'delta_g_mag.txt'), 'r') as f:
                delta_g_mag = float(f.read().strip())

        with self.detect.context():
            faces = self.detect(og)[0]
        face = faces[face_idx]

        with self.align.context():
            cna, outer_cna, ucna_coords = self.align(og, face)
        cna.save(os.path.join(dir, 'cna.png'))
        outer_cna.save(os.path.join(dir, 'outer_cna.png'))

        cna_tensor = pil_to_tensor(cna)
        with self.encode.context():
            w_plus = self.encode(cna_tensor)[0]

        with self.finetune.context():
            ft_G, ft_encs = self.finetune(cna_tensor, w_plus, DELTA_G_SIGN)

        with self.process.context():
            previews = [
                self.process(ft_encs, delta_g_mag)[0]
                for delta_g_mag in np.linspace(0, 1, N_PREVIEWS)
            ]
        create_img_row(previews, 256).save(os.path.join(dir, 'previews.png'))

        with self.inpaint.context():
            new_cna, inpainted = self.inpaint(
                outer_cna,
                w_plus,
                ft_G,
                DELTA_G_SIGN,
                delta_g_mag,
            )
        new_cna.save(os.path.join(dir, 'new_cna.png'))
        inpainted.save(os.path.join(dir, 'inpainted.png'))

        with self.unalign.context():
            final, final_wm = self.unalign(og, inpainted, ucna_coords)
        final.save(os.path.join(dir, 'final.png'))
        final_wm.save(os.path.join(dir, 'final_wm.png'))

        w, h = og.size
        comparison = Image.new('RGB', (w * 2, h), 'black')
        comparison.paste(og, (0, 0))
        comparison.paste(final, (w, 0))
        comparison.save(os.path.join(dir, 'comparison.png'))


def run_all():
    aisys = TestAisys()
    for id in sorted(os.listdir(TEST_DIR)):
        print(id)
        aisys(os.path.join(TEST_DIR, id), 'og_full.png')


def run_hero():
    aisys = TestAisys()
    aisys(os.path.join(DATA_PATH, 'hero'), 'og.jpg', 2, .8)


if __name__ == '__main__':
    run_hero()
