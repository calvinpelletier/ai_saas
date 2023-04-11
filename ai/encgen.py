import torch.nn as nn



def enc_img_gen(enc_gen, img_gen, base_enc, identity, base_latent, delta, mag):
    misc.assert_shape(base_enc, [1, 512, 4, 4])
    misc.assert_shape(identity, [1, 512, 4, 4])
    misc.assert_shape(base_latent, [1, 512])
    misc.assert_shape(delta, [1, 512])
    misc.assert_shape(mag, [1])

    # enc
    latent = base_latent + delta * torch.reshape(mag, (-1, 1))
    enc_delta = enc_gen(identity, latent)
    enc = base_enc + enc_delta * torch.reshape(mag, (-1, 1, 1, 1))

    # img
    return img_gen(enc, noise_mode='const')
