import torch
import torch.nn.functional as F
from torchvision import transforms

from ai_saas.ai.diffusion.network import Network
from ai_saas.ai.util.etc import resize


class DiffusionInpainter:
    def __init__(self, model_path):
        self.tfs = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5]),
        ])

        self.sample_num = 8

        self.netG = Network(
            unet={
                'in_channel': 6,
                'out_channel': 3,
                'inner_channel': 64,
                'channel_mults': [1, 2, 4, 8],
                'attn_res': [16],
                'num_head_channels': 32,
                'res_blocks': 2,
                'dropout': 0.2,
                'image_size': 256,
            },
            beta_schedule={
                'train': {
                    'schedule': 'linear',
                    'n_timestep': 2000,
                    'linear_start': 1e-6,
                    'linear_end': 0.01,
                },
                'test': {
                    'schedule': 'linear',
                    'n_timestep': 1000,
                    'linear_start': 1e-4,
                    'linear_end': 0.09,
                },
            },
            module_name='guided_diffusion',
            init_type='kaiming',
        )
        self.netG.load_state_dict(torch.load(model_path), strict=False)
        self.netG = self.netG.to('cuda')
        self.netG.eval()
        self.netG.set_loss(F.mse_loss)
        self.netG.set_new_noise_schedule(phase='test')

    def prep(self, img, mask):
        img = self.tfs(img).to('cuda').unsqueeze(0)

        mask = F.interpolate(
            mask.unsqueeze(0),
            size=(256, 256),
            mode='nearest',
        )

        cond_img = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        return img, cond_img, mask_img, mask

    def __call__(self, img, mask, output_size=512):
        gt_img, cond_img, mask_img, mask = self.prep(img, mask)

        output, visuals = self.netG.restoration(
            cond_img,
            y_t=cond_img,
            y_0=gt_img,
            mask=mask,
            sample_num=self.sample_num,
        )

        return resize(output, output_size)[0].float()
