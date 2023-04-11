import torch

from ai_saas.ai.model.ae.noise import NoiseAutoencoder
from ai_saas.ai.model.lerp.enc import EncLerpV1, GenOnlyEncLerpV1
from ai_saas.lib.dep import download_model
from ai_saas.lib.etc import AttrDict
from ai_saas.ai.model.lerp.latent import LatentF


def build_process_models(device, gen_only=False):
    img_model = NoiseAutoencoder(
        AttrDict({'dataset': {'imsize': 256}}),
        n_layers_per_res=[2, 2, 4, 8, 4, 2],
        conv_clamp=256,
    )
    img_model.load_state_dict(torch.load(download_model('fp/ae.pt'))['G_ema'])
    if gen_only:
        img_model = img_model.g
    img_model = img_model.to(device).eval()

    if gen_only:
        enc_model = GenOnlyEncLerpV1(EncLerpV1(
            None,
            n_id_res_blocks=4,
            n_w_res_blocks=2,
            n_delta_res_blocks=2,
            n_gen_blocks=4,
            final_conv_k=3,
            g_type='excitation',
            g_norm='none',
            g_actv='mish',
        ).enc_generator)
        sd = torch.load(download_model('fp/gen_only_f_enc.pt'))
    else:
        enc_model = EncLerpV1(
            None,
            n_id_res_blocks=4,
            n_w_res_blocks=2,
            n_delta_res_blocks=2,
            n_gen_blocks=4,
            final_conv_k=3,
            g_type='excitation',
            g_norm='none',
            g_actv='mish',
        )
        sd = torch.load(download_model('fp/f_enc.pt'))['model']
    enc_model.load_state_dict(sd)
    enc_model = enc_model.to(device).eval()

    return img_model, enc_model


def build_f_w(device):
    model = LatentF(
        levels=['coarse', 'medium', 'fine'],
        final_activation='linear',
        mult=0.1,
        lr_mul=0.01,
        pixel_norm=True,
    )
    model.load_state_dict(torch.load(download_model('fp/f_w.pt')))
    model = model.to(device).eval()
    return model
