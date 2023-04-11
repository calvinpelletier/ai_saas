import torch
import sys

from ai_saas.lib.worker import Worker
from ai_saas.ai.util.pretrained import build_pretrained_e4e
from ai_saas.ai.util.etc import pil_to_tensor, resize


class EncodeWorker(Worker):
    def __init__(self, env):
        super().__init__('encode', env, Encode())

    def pre(self, iid):
        img = self.storage.read_from_img_dir(iid, 'cna.png')
        return [pil_to_tensor(img)]

    def post(self, iid, w_plus):
        w_plus = w_plus.cpu().numpy().tolist()
        self.storage.write_to_img_dir(iid, 'w_plus.json', w_plus)


class Encode:
    def __init__(self, device='cuda'):
        self.model = build_pretrained_e4e(device=device)

    def context(self):
        return torch.no_grad()

    def __call__(self, img):
        img = resize(img, 256)
        w_plus = self.model(img)
        return [w_plus[0]]


if __name__ == '__main__':
    worker = EncodeWorker(sys.argv[1])
    worker.start()
