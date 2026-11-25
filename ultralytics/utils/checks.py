import inspect
import math
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from matplotlib import font_manager

from ultralytics.utils import LOGGER, LINUX, ROOT, clean_url, colorstr, is_colab, is_docker, is_kaggle

def is_ascii(text):
    return str(text).isascii()


def check_imgsz(imgsz, stride=32, min_dim=1, max_dim=2, floor=0):
    stride = int(stride.max() if isinstance(stride, torch.Tensor) else stride)

    if isinstance(imgsz, int):
        imgsz = [imgsz]
    elif isinstance(imgsz, (list, tuple)):
        imgsz = list(imgsz)
    else:
        raise TypeError(f"'imgsz={imgsz}' is of invalid type {type(imgsz).__name__}. "
                        f"Valid imgsz types are int i.e. 'imgsz=640' or list i.e. 'imgsz=[640,640]'")

    if len(imgsz) > max_dim:
        msg = "Use one image size for training or validation, or [height, width] for prediction."
        if max_dim != 1:
            raise ValueError(f'imgsz={imgsz} is not a valid image size. {msg}')
        LOGGER.warning(f"WARNING ⚠️ updating to 'imgsz={max(imgsz)}'. {msg}")
        imgsz = [max(imgsz)]
    sz = [max(math.ceil(x / stride) * stride, floor) for x in imgsz]

    if sz != imgsz:
        LOGGER.warning(f'WARNING ⚠️ imgsz={imgsz} must be multiple of max stride {stride}, updating to {sz}')

    sz = [sz[0], sz[0]] if min_dim == 2 and len(sz) == 1 else sz[0] if min_dim == 1 and len(sz) == 1 else sz

    return sz


def check_version(current, required, hard=False):
    def numeric(version):
        return tuple(int(n) for n in re.findall(r'\d+', version.split('+', 1)[0])[:3])

    supported = numeric(current) >= numeric(required)
    if hard and not supported:
        raise RuntimeError(f'Expected version {required} or newer, got {current}')
    return supported


def check_suffix(file, suffix='.pt', msg=''):
    if file and suffix:
        if isinstance(suffix, str):
            suffix = (suffix, )
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower().strip()
            if len(s):
                assert s in suffix, f'{msg}{f} acceptable suffix is {suffix}, not {s}'


def check_imshow(warn=False):
    try:
        if LINUX:
            assert 'DISPLAY' in os.environ and not is_docker() and not is_colab() and not is_kaggle()
        cv2.imshow('test', np.zeros((8, 8, 3), dtype=np.uint8))
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except Exception as e:
        if warn:
            LOGGER.warning(f'WARNING ⚠️ Environment does not support cv2.imshow() or PIL Image.show()\n{e}')
        return False


def print_args(args: Optional[dict] = None, show_file=True, show_func=False):

    def strip_auth(v):
        return clean_url(v) if (isinstance(v, str) and v.startswith('http') and len(v) > 100) else v

    x = inspect.currentframe().f_back
    file, _, func, _, _ = inspect.getframeinfo(x)
    if args is None:
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}
    try:
        file = Path(file).resolve().relative_to(ROOT).with_suffix('')
    except ValueError:
        file = Path(file).stem
    s = (f'{file}: ' if show_file else '') + (f'{func}: ' if show_func else '')
    LOGGER.info(colorstr(s) + ', '.join(f'{k}={strip_auth(v)}' for k, v in args.items()))


@lru_cache(maxsize=None)
def check_font(font='Arial.ttf'):
    if Path(font).is_file():
        return str(font)
    family = Path(font).stem
    for installed in font_manager.fontManager.ttflist:
        if family.lower() in installed.name.lower():
            return installed.fname
    return font_manager.findfont("DejaVu Sans")


def check_file(file, suffix='', download=False, hard=True):
    check_suffix(file, suffix)
    path = Path(file).expanduser()
    if path.is_file():
        return str(path.resolve())
    if hard:
        raise FileNotFoundError(f"File does not exist: {path}")
    return None


def check_yaml(file, suffix=('.yaml', '.yml'), hard=True):
    return check_file(file, suffix=suffix, hard=hard)
