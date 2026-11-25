import contextlib
import hashlib
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from ultralytics.nn.autobackend import check_class_names
from ultralytics.utils import LOGGER, emojis, yaml_load
from ultralytics.utils.checks import check_yaml
from ultralytics.utils.ops import segments2boxes

HELP_URL = 'See https://docs.ultralytics.com/datasets/detect for dataset formatting guidance.'
IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv', 'webm'
PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'


def img2label_paths(img_paths):
    sa, sb = f'{os.sep}images{os.sep}', f'{os.sep}labels{os.sep}'
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]


def get_hash(paths):
    size = sum(os.path.getsize(p) for p in paths if os.path.exists(p))
    h = hashlib.sha256(str(size).encode())
    h.update(''.join(paths).encode())
    return h.hexdigest()


def exif_size(img: Image.Image):
    s = img.size
    if img.format == 'JPEG':
        with contextlib.suppress(Exception):
            exif = img.getexif()
            if exif:
                rotation = exif.get(274, None)
                if rotation in [6, 8]:
                    s = s[1], s[0]
    return s


def verify_image_label(args):
    im_file, lb_file, prefix, num_cls = args
    nm, nf, ne, nc, msg, segments = 0, 0, 0, 0, '', []
    try:
        im = Image.open(im_file)
        im.verify()
        shape = exif_size(im)
        shape = (shape[1], shape[0])
        assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
        assert im.format.lower() in IMG_FORMATS, f'invalid image format {im.format}'
        if im.format.lower() in ('jpg', 'jpeg'):
            with open(im_file, 'rb') as f:
                f.seek(-2, 2)
                if f.read() != b'\xff\xd9':
                    ImageOps.exif_transpose(Image.open(im_file)).save(im_file, 'JPEG', subsampling=0, quality=100)
                    msg = f'{prefix}WARNING ⚠️ {im_file}: corrupt JPEG restored and saved'

        if os.path.isfile(lb_file):
            nf = 1
            with open(lb_file) as f:
                lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                if any(len(x) > 6 for x in lb):
                    classes = np.array([x[0] for x in lb], dtype=np.float32)
                    segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]
                    lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)
                lb = np.array(lb, dtype=np.float32)
            nl = len(lb)
            if nl:
                assert lb.shape[1] == 5, f'labels require 5 columns, {lb.shape[1]} columns detected'
                points = lb[:, 1:]
                assert points.max() <= 1, f'non-normalized or out of bounds coordinates {points[points > 1]}'
                assert lb.min() >= 0, f'negative label values {lb[lb < 0]}'

                max_cls = lb[:, 0].max()
                assert max_cls < num_cls, \
                    f'Label class {int(max_cls)} exceeds dataset class count {num_cls}. ' \
                    f'Possible class labels are 0-{num_cls - 1}'
                _, i = np.unique(lb, axis=0, return_index=True)
                if len(i) < nl:
                    lb = lb[i]
                    if segments:
                        segments = [segments[x] for x in i]
                    msg = f'{prefix}WARNING ⚠️ {im_file}: {nl - len(i)} duplicate labels removed'
            else:
                ne = 1
                lb = np.zeros((0, 5), dtype=np.float32)
        else:
            nm = 1
            lb = np.zeros((0, 5), dtype=np.float32)
        return im_file, lb, shape, segments, nm, nf, ne, nc, msg
    except Exception as e:
        nc = 1
        msg = f'{prefix}WARNING ⚠️ {im_file}: ignoring corrupt image/label: {e}'
        return [None, None, None, None, nm, nf, ne, nc, msg]


def check_det_dataset(dataset):
    yaml_file = Path(check_yaml(dataset))
    data = yaml_load(yaml_file, append_filename=True)

    for k in 'train', 'val':
        if k not in data:
            if k == 'val' and 'validation' in data:
                LOGGER.info("WARNING ⚠️ renaming data YAML 'validation' key to 'val' to match YOLO format.")
                data['val'] = data.pop('validation')
            else:
                raise SyntaxError(
                    emojis(f"{dataset} '{k}:' key missing ❌.\n'train' and 'val' are required in all data YAMLs."))
    if 'names' not in data and 'nc' not in data:
        raise SyntaxError(emojis(f"{dataset} key missing ❌.\n either 'names' or 'nc' are required in all data YAMLs."))
    if 'names' in data and 'nc' in data and len(data['names']) != data['nc']:
        raise SyntaxError(emojis(f"{dataset} 'names' length {len(data['names'])} and 'nc: {data['nc']}' must match."))
    if 'names' not in data:
        data['names'] = [f'class_{i}' for i in range(data['nc'])]
    else:
        data['nc'] = len(data['names'])

    data['names'] = check_class_names(data['names'])

    path = (yaml_file.parent / (data.get('path') or '.')).resolve()
    data['path'] = path
    for k in 'train', 'val', 'test':
        if data.get(k):
            if isinstance(data[k], str):
                x = (path / data[k]).resolve()
                if not x.exists() and data[k].startswith('../'):
                    x = (path / data[k][3:]).resolve()
                data[k] = str(x)
            else:
                data[k] = [str((path / x).resolve()) for x in data[k]]

    for split in ("train", "val"):
        paths = data[split] if isinstance(data[split], list) else [data[split]]
        for split_path in paths:
            if not Path(split_path).exists():
                raise FileNotFoundError(f"Missing {split} path in {dataset}: {split_path}")
    return data
