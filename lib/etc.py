import os
import string
import random


def get_environ(name):
    val = os.environ.get(name)
    if val is None:
        raise Exception(f'missing {name} env var')
    return val


def gen_key(length=16):
    return ''.join(random.choices(
        string.ascii_letters + string.digits,
        k=length,
    ))


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self
        for key, value in self.__dict__.items():
            if isinstance(value, dict):
                self.__dict__[key] = AttrDict(value)
            elif isinstance(value, (list, tuple)):
                if len(value) > 0 and isinstance(value[0], dict):
                    self.__dict__[key] = [AttrDict(item) for item in value]
                else:
                    self.__dict__[key] = value
