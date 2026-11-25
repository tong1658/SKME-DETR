import contextlib
import logging.config
import os
import platform
import re
import sys
import tempfile
import threading
import urllib.parse
from pathlib import Path
from types import SimpleNamespace
from typing import Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from tqdm import tqdm as tqdm_original

RANK = int(os.getenv('RANK', -1))
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG_PATH = ROOT / 'cfg/default.yaml'
NUM_THREADS = min(8, max(1, (os.cpu_count() or 1) - 1))
VERBOSE = str(os.getenv('YOLO_VERBOSE', True)).lower() == 'true'
TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}' if VERBOSE else None
LOGGING_NAME = 'ultralytics'
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ['Darwin', 'Linux', 'Windows'])

torch.set_printoptions(linewidth=320, precision=4, profile='default')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})
cv2.setNumThreads(0)
os.environ['NUMEXPR_MAX_THREADS'] = str(NUM_THREADS)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


class TQDM(tqdm_original):

    def __init__(self, *args, **kwargs):
        kwargs['disable'] = not VERBOSE or kwargs.get('disable', False)
        kwargs.setdefault('bar_format', TQDM_BAR_FORMAT)
        super().__init__(*args, **kwargs)


class SimpleClass:

    def __str__(self):
        attr = []
        for a in dir(self):
            v = getattr(self, a)
            if not callable(v) and not a.startswith('_'):
                if isinstance(v, SimpleClass):
                    s = f'{a}: {v.__module__}.{v.__class__.__name__} object'
                else:
                    s = f'{a}: {repr(v)}'
                attr.append(s)
        return f'{self.__module__}.{self.__class__.__name__} object with attributes:\n\n' + '\n'.join(attr)

    def __repr__(self):
        return self.__str__()

    def __getattr__(self, attr):
        name = self.__class__.__name__
        raise AttributeError(f"'{name}' object has no attribute '{attr}'")


class IterableSimpleNamespace(SimpleNamespace):

    def __iter__(self):
        return iter(vars(self).items())

    def __str__(self):
        return '\n'.join(f'{k}={v}' for k, v in vars(self).items())

    def __getattr__(self, attr):
        name = self.__class__.__name__
        raise AttributeError(f"{name} has no setting {attr!r}. See {DEFAULT_CFG_PATH} for supported settings.")

    def get(self, key, default=None):
        return getattr(self, key, default)


def plt_settings(rcparams=None, backend='Agg'):

    if rcparams is None:
        rcparams = {'font.size': 11}

    def decorator(func):

        def wrapper(*args, **kwargs):
            original_backend = plt.get_backend()
            if backend != original_backend:
                plt.close('all')
                plt.switch_backend(backend)

            with plt.rc_context(rcparams):
                result = func(*args, **kwargs)

            if backend != original_backend:
                plt.close('all')
                plt.switch_backend(original_backend)
            return result

        return wrapper

    return decorator


def set_logging(name=LOGGING_NAME, verbose=True):
    rank = int(os.getenv('RANK', -1))
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            name: {
                'format': '%(message)s'}},
        'handlers': {
            name: {
                'class': 'logging.StreamHandler',
                'formatter': name,
                'level': level}},
        'loggers': {
            name: {
                'level': level,
                'handlers': [name],
                'propagate': False}}})


def emojis(string=''):
    if WINDOWS:
        encoding = getattr(sys.stderr, 'encoding', None) or 'utf-8'
        return str(string).encode(encoding, errors='replace').decode(encoding)
    return string


class EmojiFilter(logging.Filter):

    def filter(self, record):
        record.msg = emojis(record.getMessage())
        record.args = ()
        return super().filter(record)


set_logging(LOGGING_NAME, verbose=VERBOSE)
LOGGER = logging.getLogger(LOGGING_NAME)
if WINDOWS:
    LOGGER.addFilter(EmojiFilter())


def yaml_save(file='data.yaml', data=None, header=''):
    if data is None:
        data = {}
    file = Path(file)
    if not file.parent.exists():
        file.parent.mkdir(parents=True, exist_ok=True)

    valid_types = int, float, str, bool, list, tuple, dict, type(None)
    for k, v in data.items():
        if not isinstance(v, valid_types):
            data[k] = str(v)

    with open(file, 'w', errors='ignore', encoding='utf-8') as f:
        if header:
            f.write(header)
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def yaml_load(file='data.yaml', append_filename=False):
    assert Path(file).suffix in ('.yaml', '.yml'), f'Attempting to load non-YAML file {file} with yaml_load()'
    with open(file, errors='ignore', encoding='utf-8') as f:
        s = f.read()

        if not s.isprintable():
            s = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+', '', s)

        data = yaml.safe_load(s) or {}
        if append_filename:
            data['yaml_file'] = str(file)
        return data


DEFAULT_CFG_DICT = yaml_load(DEFAULT_CFG_PATH)
for k, v in DEFAULT_CFG_DICT.items():
    if isinstance(v, str) and v.lower() == 'none':
        DEFAULT_CFG_DICT[k] = None
DEFAULT_CFG_KEYS = DEFAULT_CFG_DICT.keys()
DEFAULT_CFG = IterableSimpleNamespace(**DEFAULT_CFG_DICT)


def is_colab():
    return 'COLAB_RELEASE_TAG' in os.environ or 'COLAB_BACKEND_VERSION' in os.environ


def is_kaggle():
    return os.environ.get('PWD') == '/kaggle/working' and os.environ.get('KAGGLE_URL_BASE') == 'https://www.kaggle.com'


def is_docker() -> bool:
    file = Path('/proc/self/cgroup')
    if file.exists():
        with open(file) as f:
            return 'docker' in f.read()
    else:
        return False


def is_dir_writeable(dir_path: Union[str, Path]) -> bool:
    return os.access(str(dir_path), os.W_OK)


USER_CONFIG_DIR = Path(tempfile.gettempdir()) / 'rtdetr'

def colorstr(*input):
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])
    colors = {
        'black': '\033[30m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


class TryExcept(contextlib.ContextDecorator):

    def __init__(self, msg='', verbose=True):
        self.msg = msg
        self.verbose = verbose

    def __enter__(self):
        pass

    def __exit__(self, exc_type, value, traceback):
        if self.verbose and value:
            print(emojis(f"{self.msg}{': ' if self.msg else ''}{value}"))
        return True


def threaded(func):

    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs, daemon=True)
        thread.start()
        return thread

    return wrapper


def clean_url(url):
    url = Path(url).as_posix().replace(':/', '://')
    return urllib.parse.unquote(url).split('?')[0]


RUNS_DIR = ROOT.parent / 'runs'

from .patches import imread, imshow, imwrite

if WINDOWS:
    cv2.imread, cv2.imwrite, cv2.imshow = imread, imwrite, imshow
