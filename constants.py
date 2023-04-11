import os
from enum import IntEnum

from ai_saas.lib.etc import get_environ


class Status(IntEnum):
    ERROR = -1
    NOT_STARTED = 0
    IN_PROGRESS = 1
    DONE = 2


CODE_PATH = os.path.join(get_environ('BASE_CODE_PATH'), 'fp')
DATA_PATH = os.path.join(get_environ('BASE_DATA_PATH'), 'fp')
CONFIG_PATH = os.path.join(CODE_PATH, 'config')

IMG_DATA_DIR = 'img-data'

THUMBNAIL_SIZE = 256

ORIGINAL_IMG = 'og_full.png'
FINAL_IMG = 'final_full.png'
FINAL_IMG_WM = 'final_full_wm.png'
FINAL_THUMB = 'final_thumb.png'
