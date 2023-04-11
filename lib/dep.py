from google.cloud import storage
import os

from ai_saas.constants import DATA_PATH


client = storage.Client()
bucket = client.bucket('fp-dep-bucket')


def download_file(path):
    dest_path = os.path.join(DATA_PATH, path)
    if not os.path.exists(dest_path):
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        blob = bucket.blob(path)
        blob.download_to_filename(dest_path)
    return dest_path


def download_model(path):
    return download_file(f'models/{path}')


def download_img(path):
    return download_file(f'imgs/{path}')
