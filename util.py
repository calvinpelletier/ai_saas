from PIL import Image

from ai_saas.constants import THUMBNAIL_SIZE


def create_thumbnail(img):
    scale_factor = THUMBNAIL_SIZE / max(img.size)
    return img.resize((
        int(scale_factor * img.size[0]),
        int(scale_factor * img.size[1]),
    ), Image.ANTIALIAS)
