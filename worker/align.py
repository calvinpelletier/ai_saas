import dlib
from contextlib import nullcontext
import numpy as np
import sys

from ai_saas.constants import ORIGINAL_IMG
from ai_saas.lib.worker import Worker
from ai_saas.lib.dep import download_model
from ai_saas.ai.cna import crop_and_align, unalign_coords, align_coords
from ai_saas.ai.face import predict_landmarks, landmarks_to_alignment_coords
from ai_saas.ai.constants import CNA_SIZE, OUTER_CNA_SIZE, ALIGN_TRANSFORM_SIZE, \
    OUTER_COORDS_ALIGNED


class AlignWorker(Worker):
    def __init__(self, env):
        super().__init__('align', env, Align(), needs_db=True)

    def pre(self, iid):
        img = self.storage.read_from_img_dir(iid, ORIGINAL_IMG)
        face = self.db.get_image(iid)['image']['si']['face']
        return img, face

    def post(self, iid, cna, outer_cna, ucna_coords):
        self.storage.write_to_img_dir(iid, 'cna.png', cna)
        self.storage.write_to_img_dir(iid, 'outer_cna.png', outer_cna)
        self.storage.write_to_img_dir(iid, 'ucna_coords.json', ucna_coords)


class Align:
    def __init__(self):
        self.landmark_predictor = dlib.shape_predictor(
            download_model('dlib/landmark.dat'),
        )

    def context(self):
        return nullcontext()

    def __call__(self, img, face):
        landmarks = predict_landmarks(self.landmark_predictor, img, face)

        inner_coords = landmarks_to_alignment_coords(landmarks)
        cna = crop_and_align(img, inner_coords, CNA_SIZE, ALIGN_TRANSFORM_SIZE)

        outer_coords = np.array(unalign_coords(
            OUTER_COORDS_ALIGNED,
            [int(round(x)) for x in list(inner_coords)],
            ALIGN_TRANSFORM_SIZE,
        ), dtype=np.float32)
        outer_cna = crop_and_align(
            img,
            outer_coords,
            OUTER_CNA_SIZE,
            ALIGN_TRANSFORM_SIZE,
        )

        w, h = img.size
        ucna_coords = align_coords(
            (
                0, 0,  # nw
                0, h,  # sw
                w, h,  # se
                w, 0,  # ne
            ),
            outer_coords,
            ALIGN_TRANSFORM_SIZE,
        )
        ucna_coords = [int(round(x)) for x in list(ucna_coords)]

        return cna, outer_cna, ucna_coords


if __name__ == '__main__':
    worker = AlignWorker(sys.argv[1])
    worker.start()
