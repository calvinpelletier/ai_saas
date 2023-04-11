import dlib
from contextlib import nullcontext
import sys

from ai_saas.constants import ORIGINAL_IMG
from ai_saas.ai.face import detect_faces
from ai_saas.lib.worker import Worker


class DetectWorker(Worker):
    def __init__(self, env):
        super().__init__('detect', env, Detect(), needs_db=True)

    def pre(self, iid):
        img = self.storage.read_from_img_dir(iid, ORIGINAL_IMG)
        return [img]

    def post(self, iid, faces):
        resp = self.db.set_image_attr(iid, 'faces', faces)
        assert resp['success']


class Detect:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()

    def context(self):
        return nullcontext()

    def __call__(self, img):
        faces = detect_faces(self.detector, img)
        return [faces]


if __name__ == '__main__':
    worker = DetectWorker(sys.argv[1])
    worker.start()
