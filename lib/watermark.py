from PIL import Image, ImageDraw, ImageFont
import os
from ai_saas.lib.dep import download_img


# TODO
WATERMARK_IMG = Image.open(download_img('watermark.png'))


# TODO
def add_watermark(img):
    ret = img.copy().convert('RGBA')
    w, h = ret.size
    wm_size = int(w * 0.1)
    watermark = WATERMARK_IMG.resize((wm_size, wm_size))
    # watermark.putalpha(127)
    wm_data = watermark.load()
    for y in range(wm_size):
        for x in range(wm_size):
            if wm_data[x, y] == (255, 255, 255, 255):
                wm_data[x, y] = (255, 255, 255, 127)
    watermark_full = Image.new('RGBA', ret.size, (0, 0, 0, 0))
    watermark_full.paste(watermark, (w // 2, h // 2))
    return Image.alpha_composite(ret, watermark_full)
