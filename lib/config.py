import yaml
import os
from io import StringIO

from ai_saas.lib.etc import AttrDict
from ai_saas.constants import CONFIG_PATH


def load_config(name, env):
    cat_f = StringIO()
    with open(os.path.join(CONFIG_PATH, f'{name}.yaml'), 'r') as f:
        cat_f.write(f.read())
    with open(os.path.join(CONFIG_PATH, f'env/{env}.yaml'), 'r') as f:
        cat_f.write(f.read())
    cat_f.seek(0)
    cfg = AttrDict(**yaml.safe_load(cat_f))
    return cfg
