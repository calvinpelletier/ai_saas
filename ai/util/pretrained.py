#!/usr/bin/env python3
import torch
import os
import argparse

from ai_saas.ai.sg2.model import Generator
from ai_saas.ai.e4e.models.encoders.psp_encoders import Encoder4Editing
from ai_saas.ai.model.facegen.gen_for_seg import GeneratorForSeg
from ai_saas.ai.model.facegen.tri import TriGenerator
from ai_saas.lib.dep import download_model


def build_pretrained_sg2(
    eval=True,
    device='cuda',
    g_type='reg',
    path_override=None,
    load_weights=True,
):
    if g_type == 'reg':
        g_module = Generator
    elif g_type == 'seg':
        g_module = GeneratorForSeg
    elif g_type == 'tri':
        g_module = TriGenerator
    else:
        raise Exception(g_type)

    model = g_module(
        512, # z dims
        0, # c dims
        512, # w dims
        1024, # imsize
        3, # channels
        mapping_kwargs={
            'num_layers': 8,
        },
        synthesis_kwargs={
            'channel_base': 32768,
            'channel_max': 512,
            'num_fp16_res': 4,
            'conv_clamp': 256,
        },
    )

    if load_weights:
        if path_override is None:
            path = download_model('sg2/official_g.pt')
        else:
            path = path_override
        model.load_state_dict(torch.load(path))
    if eval:
        model = model.eval().requires_grad_(False)
    model = model.to(device)
    return model


def build_pretrained_e4e(eval=True, device='cuda', load_weights=True):
    path = download_model('e4e/e4e_ffhq_encode.pt')
    ckpt = torch.load(path)
    opts = argparse.Namespace(**ckpt['opts'])

    e4e = Encoder4Editing(50, 'ir_se', opts)

    if load_weights:
        e4e_dict = {k.replace('encoder.', ''): v \
            for k, v in ckpt['state_dict'].items() if k.startswith('encoder.')}
        e4e.load_state_dict(e4e_dict)

    if eval:
        e4e.eval()
    e4e = e4e.to(device)

    latent_avg = ckpt['latent_avg'].to(device)
    def add_latent_avg(model, inputs, outputs):
        return outputs + latent_avg.repeat(outputs.shape[0], 1, 1)
    e4e.register_forward_hook(add_latent_avg)

    return e4e
