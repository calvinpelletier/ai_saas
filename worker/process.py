import torch
import numpy as np
import sys

from ai_saas.lib.worker import LiveWorker
from ai_saas.ai.util.etc import pil_to_tensor, resize, tensor_to_pil
from ai_saas.ai.constants import MAX_MAG
from ai_saas.lib.ai import build_process_models
from ai_saas.lib.img import encode_img
from ai_saas.lib.cache import lru_cache


def convert_enc(enc, device):
    return torch.from_numpy(np.asarray(
        enc,
    )).to(torch.float32).to(device).unsqueeze(0)


class ProcessWorker(LiveWorker):
    def __init__(self, env):
        super().__init__('process', env, Process())
        self.device = self.cfg.task.device

    def pre(self, req):
        encs = self.load_encs(req['iid'])
        delta_g_mag = float(req['di']['delta_g_mag'])
        return encs, delta_g_mag

    def post(self, img):
        return encode_img(img)

    @lru_cache(maxsize=256)
    def load_encs(self, iid):
        encs = self.storage.read_from_img_dir(iid, 'ft_encs.json')
        encs = {k: convert_enc(v, self.device) for k, v in encs.items()}
        return encs


class Process:
    def __init__(self, device='cuda'):
        self.device = device
        self.G_img, self.G_enc = build_process_models(
            device,
            gen_only=True,
        )

    def context(self):
        return torch.no_grad()

    def __call__(self, encs, delta_g_mag):
        mag = torch.tensor(
            delta_g_mag * MAX_MAG,
        ).unsqueeze(0).to(torch.float32).to(self.device)

        preview = self.G_img(self.G_enc(
            encs['base_enc'],
            encs['identity'],
            encs['base_latent'],
            encs['delta'],
            mag,
        ), noise_mode='const')

        return [tensor_to_pil(preview)]


if __name__ == '__main__':
    worker = ProcessWorker(sys.argv[1])
    worker.start()
