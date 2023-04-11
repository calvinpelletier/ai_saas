from PIL import Image
from base64 import b64decode, encodebytes
from io import BytesIO


def encode_img(img):
    byte_arr = BytesIO()
    img.save(byte_arr, format='PNG')
    return encodebytes(byte_arr.getvalue()).decode('ascii')


def decode_img(data):
    return Image.open(BytesIO(b64decode(data)))
