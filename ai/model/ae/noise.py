#!/usr/bin/env python3
import torch.nn as nn
from copy import deepcopy
import ai_saas.ai.sg2.misc as misc
from ai_saas.ai.util.etc import log2_diff
from ai_saas.ai.blocks.res import FancyMultiLayerDownBlock, ClampResDownConvBlock, \
    ResDownConvBlock, NoiseResUpConvBlock
from ai_saas.ai.blocks.conv import ConvBlock, CustomConvBlock, ConvToImg
from ai_saas.ai.blocks.encode import FeatMapToLatentViaFc
from ai_saas.ai.blocks.quant import SimpleNoiseResUpConvBlock


class NoiseResDecoder(nn.Module):
    def __init__(self,
        imsize=256,
        smallest_imsize=4,
        nc_in=3,
        nc_base=64,
        nc_max=512,
        norm='batch',
        actv='mish',
        conv_clamp=None,
        onnx=False,
        simple=False,
    ):
        super().__init__()
        self.n_blocks = log2_diff(imsize, smallest_imsize)
        nc = [min(nc_max, nc_base * (2 ** i)) for i in range(self.n_blocks + 1)]
        nc = nc[::-1]

        for i in range(self.n_blocks):
            if simple:
                block = SimpleNoiseResUpConvBlock(
                    smallest_imsize * (2 ** (i + 1)),
                    nc[i],
                    nc[i+1],
                )
            else:
                block = NoiseResUpConvBlock(
                    smallest_imsize * (2 ** (i + 1)),
                    nc[i],
                    nc[i+1],
                    norm=norm,
                    actv=actv,
                    conv_clamp=conv_clamp,
                    onnx=onnx,
                )
            setattr(self, f'b{i}', block)

        self.final = ConvToImg(nc[-1])

    def forward(self, x, noise_mode='random'):
        for i in range(self.n_blocks):
            x = getattr(self, f'b{i}')(x, noise_mode=noise_mode)
        return self.final(x)

    def fuse_model(self):
        for i in range(self.n_blocks):
            getattr(self, f'b{i}').fuse_model()


class SqueezeEncoder(nn.Module):
    def __init__(self,
        input_imsize=128,
        smallest_imsize=4,
        nc_in=3,
        nc_base=32,
        nc_max=512,
        n_layers_per_res=[2, 4, 8, 4, 2],
        norm='batch',
        actv='mish',
        conv_clamp=None,
        dropout_after_squeeze_layers=False,
    ):
        super().__init__()

        n_down_up = log2_diff(input_imsize, smallest_imsize)
        assert len(n_layers_per_res) == n_down_up
        nc = [min(nc_max, nc_base * (2 ** i)) for i in range(n_down_up + 1)]

        # initial block
        if conv_clamp is not None:
            blocks = [CustomConvBlock(
                nc_in,
                nc[0],
                norm='none',
                actv=actv,
                conv_clamp=conv_clamp,
            )]
        else:
            blocks = [ConvBlock(
                nc_in,
                nc[0],
                norm='none',
                actv=actv,
            )]

        # main blocks
        for i in range(n_down_up):
            if n_layers_per_res[i] == 0:
                if dropout_after_squeeze_layers and n_layers_per_res[i-1] != 0:
                    blocks.append(nn.Dropout(0.1))
                if conv_clamp is not None:
                    down = ClampResDownConvBlock(
                        nc[i],
                        nc[i+1],
                        k_down=3,
                        norm=norm,
                        weight_norm=False,
                        actv=actv,
                        conv_clamp=conv_clamp,
                    )
                else:
                    down = ResDownConvBlock(
                        nc[i],
                        nc[i+1],
                        k_down=3,
                        norm=norm,
                        weight_norm=False,
                        actv=actv,
                    )
            else:
                down = FancyMultiLayerDownBlock(
                    nc[i],
                    nc[i+1],
                    n_layers=n_layers_per_res[i],
                    norm=norm,
                    actv=actv,
                    conv_clamp=conv_clamp,
                )
            blocks.append(down)
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


def dynamic_to_learned_const(ae, img):
    assert isinstance(ae, NoiseAutoencoder)
    misc.assert_shape(img, [None, 3, ae.imsize, ae.imsize])
    enc = ae.e(img)
    ae_lc = EncLearnedConst(enc)
    return ae_lc


class EncLearnedConst(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = nn.Parameter(enc.clone().detach())

    def forward(self):
        return self.enc


class NoiseAutoencoder(nn.Module):
    def __init__(self,
        cfg,
        e_type='squeeze',
        smallest_imsize=4,
        nc_in=3,
        nc_base=64,
        nc_max=512,
        n_layers_per_res=[2, 2, 4, 8, 4, 2],
        norm='batch',
        actv='mish',
        conv_clamp=None,
        dropout_after_squeeze_layers=False,
        onnx=False,
        simple=False,
    ):
        super().__init__()
        self.intermediate = 'enc'
        self.imsize = cfg.dataset.imsize

        if e_type == 'squeeze':
            self.e = SqueezeEncoder(
                input_imsize=self.imsize,
                smallest_imsize=smallest_imsize,
                nc_in=nc_in,
                nc_base=nc_base,
                nc_max=nc_max,
                n_layers_per_res=n_layers_per_res,
                norm=norm,
                actv=actv,
                conv_clamp=conv_clamp,
                dropout_after_squeeze_layers=dropout_after_squeeze_layers,
            )
        else:
            raise Exception(e_type)

        self.g = NoiseResDecoder(
            imsize=self.imsize,
            smallest_imsize=smallest_imsize,
            nc_in=nc_in,
            nc_base=nc_base,
            nc_max=nc_max,
            norm=norm,
            actv=actv,
            conv_clamp=conv_clamp,
            onnx=onnx,
            simple=simple,
        )


    def forward(self, x, noise_mode='random'):
        encoding = self.e(x)
        return self.g(encoding, noise_mode=noise_mode)

    def prep_for_train_phase(self):
        self.requires_grad_(True)
