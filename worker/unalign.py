from contextlib import nullcontext
import sys

from ai_saas.lib.worker import Worker
from ai_saas.ai.constants import ALIGN_TRANSFORM_SIZE
from ai_saas.constants import ORIGINAL_IMG, FINAL_IMG, FINAL_IMG_WM, FINAL_THUMB
from ai_saas.ai.cna import unalign_and_uncrop
from ai_saas.lib.watermark import add_watermark
from ai_saas.util import create_thumbnail


class UnalignWorker(Worker):
    def __init__(self, env):
        super().__init__('unalign', env, Unalign())

    def pre(self, iid):
        og_img = self.storage.read_from_img_dir(iid, ORIGINAL_IMG)
        inpainted = self.storage.read_from_img_dir(iid, 'inpainted.png')
        ucna_coords = self.storage.read_from_img_dir(iid, 'ucna_coords.json')
        return og_img, inpainted, ucna_coords

    def post(self, iid, final, final_wm):
        self.storage.write_to_img_dir(iid, FINAL_IMG, final)
        self.storage.write_to_img_dir(iid, FINAL_THUMB, create_thumbnail(final))
        self.storage.write_to_img_dir(iid, FINAL_IMG_WM, final_wm)


class Unalign:
    def context(self):
        return nullcontext()

    def __call__(self, og_img, inpainted, ucna_coords):
        final = unalign_and_uncrop(
            og_img,
            ucna_coords,
            inpainted,
            ALIGN_TRANSFORM_SIZE,
        )
        final_wm = add_watermark(final)
        return final, final_wm


if __name__ == '__main__':
    worker = UnalignWorker(sys.argv[1])
    worker.start()
