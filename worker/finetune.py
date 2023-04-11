import torch
import numpy as np
from contextlib import nullcontext
from io import BytesIO
import sys

from ai_saas.lib.worker import Worker
from ai_saas.server.db.client import DatabaseClient
from ai_saas.ai.constants import MAX_MAG, FT_N_MAGS
from ai_saas.ai.pti import PtiTrainer
from ai_saas.ai.finetune.enc_lerp import finetune_enc_lerp
from ai_saas.ai.finetune.ae import finetune_ae
from ai_saas.ai.util.etc import pil_to_tensor, resize
from ai_saas.lib.ai import build_process_models, build_f_w


def export_enc(tensor):
    return tensor.squeeze(0).detach().cpu().numpy().tolist()


class FinetuneWorker(Worker):
    def __init__(self, env):
        super().__init__('finetune', env, Finetune(), needs_db=True)
        self.device = self.cfg.task.device

    def pre(self, iid):
        cna = pil_to_tensor(self.storage.read_from_img_dir(iid, 'cna.png'))
        w = torch.from_numpy(np.asarray(
            self.storage.read_from_img_dir(iid, 'w_plus.json'),
        )).to(torch.float32).to(self.device)
        delta_g_sign = torch.tensor(
            self.db.get_image(iid)['image']['si']['delta_g_sign'],
        ).to(torch.float32).to(self.device)
        return cna, w, delta_g_sign

    def post(self, iid, G, encs):
        encs = {k: export_enc(v) for k, v in encs.items()}
        self.storage.write_to_img_dir(iid, 'ft_encs.json', encs)

        bytes = BytesIO()
        torch.save(G.state_dict(), bytes)
        self.storage.write_to_img_dir(iid, 'ft_g.pt', bytes)


class Finetune:
    def __init__(self, device='cuda'):
        self.pti_trainer = PtiTrainer(device)
        self.F_w = build_f_w(device)
        self.AE, self.F_enc = build_process_models(device)
        self.mags = torch.tensor(
            np.linspace(0., MAX_MAG, num=FT_N_MAGS),
            device=device,
        ).to(torch.float32)

    def context(self):
        return nullcontext()

    def __call__(self, cna, w, delta_g_sign):
        G = self.pti_trainer.train(cna, w.unsqueeze(0)).synthesis

        with torch.no_grad():
            delta_g = delta_g_sign * self.mags
            target_imgs = resize(G(self.F_w(
                w.repeat(FT_N_MAGS, 1, 1),
                delta_g,
            ), noise_mode='const'), 256)

        AE_lc = finetune_ae(
            self.AE,
            target_imgs,
        )
        with torch.no_grad():
            target_enc = AE_lc()
            base_enc = target_enc[0, :, :, :].unsqueeze(0)
            guide_enc = target_enc[FT_N_MAGS - 1, :, :, :].unsqueeze(0)

        F_enc_lc, F_enc_g = finetune_enc_lerp(
            self.F_enc,
            base_enc,
            guide_enc,
            target_enc,
            self.mags,
        )
        with torch.no_grad():
            identity, base_latent, delta = F_enc_lc()

        return G, {
            'base_enc': base_enc,
            'identity': identity,
            'base_latent': base_latent,
            'delta': delta,
        }


if __name__ == '__main__':
    worker = FinetuneWorker(sys.argv[1])
    worker.start()
