import os
from google.cloud import storage
from io import BytesIO
from PIL import Image
import json
import datetime

from ai_saas.constants import IMG_DATA_DIR
from ai_saas.lib.etc import gen_key


class StorageClient:
    def __init__(self, bucket):
        self._client = storage.Client()
        self._bucket = self._client.bucket(bucket)

    def read(self, path):
        blob = self._bucket.blob(path)
        if not blob.exists():
            return None
        ext = path.split('.')[-1]
        return EXT_TO_FN[ext][0](blob)

    def write(self, path, value):
        blob = self._bucket.blob(path)
        ext = path.split('.')[-1]
        EXT_TO_FN[ext][1](blob, value)

    def gen_signed_url(self, path):
        blob = self._bucket.blob(path)
        if not blob.exists():
            return None
        return blob.generate_signed_url(
            version='v4',
            expiration=datetime.timedelta(minutes=10),
            method='GET',
        )

    def read_from_img_dir(self, iid, path):
        return self.read(os.path.join(IMG_DATA_DIR, str(iid), path))

    def write_to_img_dir(self, iid, path, img):
        self.write(os.path.join(IMG_DATA_DIR, str(iid), path), img)

    def signed_url_in_img_dir(self, iid, path):
        return self.gen_signed_url(os.path.join(IMG_DATA_DIR, str(iid), path))

    def read_or_init_key(self, name):
        path = f'keys/{name}.key'
        key = self.read(path)
        if key is None:
            key = gen_key()
            self.write(path, key)
        return key


def _read_string(blob):
    return blob.download_as_string().decode('utf-8')

def _write_string(blob, text):
    return blob.upload_from_string(text)


def _read_json(blob):
    return json.loads(blob.download_as_string())

def _write_json(blob, obj):
    return blob.upload_from_string(json.dumps(obj))


def _read_image(blob):
    return Image.open(BytesIO(blob.download_as_bytes()))

def _write_image(blob, img):
    f = BytesIO()
    img.save(f, format='PNG')
    f.seek(0)
    return blob.upload_from_file(f, content_type='image/png')


def _read_bytes(blob):
    return BytesIO(blob.download_as_bytes())

def _write_bytes(blob, f):
    f.seek(0)
    return blob.upload_from_file(f)


EXT_TO_FN = {
    'key': (_read_string, _write_string),
    'csv': (_read_string, _write_string),
    'png': (_read_image, _write_image),
    'json': (_read_json, _write_json),
    'pt': (_read_bytes, _write_bytes),
}
