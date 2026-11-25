from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import LOCAL_RANK, NUM_THREADS, TQDM, is_dir_writeable

from ultralytics.utils.instance import Instances
from .base import BaseDataset
from .utils import HELP_URL, LOGGER, get_hash, img2label_paths, verify_image_label

DATASET_CACHE_VERSION = '1.0.3'


class YOLODataset(BaseDataset):

    def __init__(self, *args, data=None, **kwargs):
        self.data = data
        super().__init__(*args, **kwargs)

    def cache_labels(self, path=Path('./labels.cache')):
        x = {'labels': []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []
        desc = f'{self.prefix}Scanning {path.parent / path.stem}...'
        total = len(self.im_files)
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(func=verify_image_label,
                                iterable=zip(self.im_files, self.label_files, repeat(self.prefix),
                                             repeat(len(self.data['names']))))
            pbar = TQDM(results, desc=desc, total=total)
            for im_file, lb, shape, segments, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x['labels'].append(
                        dict(
                            im_file=im_file,
                            shape=shape,
                            cls=lb[:, 0:1],
                            bboxes=lb[:, 1:],
                            segments=segments,
                            normalized=True,
                            bbox_format='xywh'))
                if msg:
                    msgs.append(msg)
                pbar.desc = f'{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt'
            pbar.close()

        if msgs:
            LOGGER.info('\n'.join(msgs))
        if nf == 0:
            LOGGER.warning(f'{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}')
        x['hash'] = get_hash(self.label_files + self.im_files)
        x['results'] = nf, nm, ne, nc, len(self.im_files)
        x['msgs'] = msgs
        save_dataset_cache_file(self.prefix, path, x)
        return x

    def get_labels(self):
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix('.cache')
        try:
            cache, exists = load_dataset_cache_file(cache_path), True
            assert cache['version'] == DATASET_CACHE_VERSION
            assert cache['hash'] == get_hash(self.label_files + self.im_files)
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False

        nf, nm, ne, nc, n = cache.pop('results')
        if exists and LOCAL_RANK in (-1, 0):
            d = f'Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt'
            TQDM(None, desc=self.prefix + d, total=n, initial=n)
            if cache['msgs']:
                LOGGER.info('\n'.join(cache['msgs']))

        [cache.pop(k) for k in ('hash', 'version', 'msgs')]
        labels = cache['labels']
        if not labels:
            LOGGER.warning(f'WARNING ⚠️ No images found in {cache_path}, training may not work correctly. {HELP_URL}')
        self.im_files = [lb['im_file'] for lb in labels]

        lengths = ((len(lb['cls']), len(lb['bboxes']), len(lb['segments'])) for lb in labels)
        len_cls, len_boxes, len_segments = (sum(x) for x in zip(*lengths))
        if len_segments and len_boxes != len_segments:
            LOGGER.warning(
                f'WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = {len_segments}, '
                f'len(boxes) = {len_boxes}. To resolve this only boxes will be used and all segments will be removed. '
                'To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset.')
            for lb in labels:
                lb['segments'] = []
        if len_cls == 0:
            LOGGER.warning(f'WARNING ⚠️ No labels found in {cache_path}, training may not work correctly. {HELP_URL}')
        return labels

    def close_mosaic(self, hyp):
        hyp.mosaic = 0.0
        hyp.copy_paste = 0.0
        hyp.mixup = 0.0
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label):
        bboxes = label.pop('bboxes')
        segments = label.pop('segments')
        label.pop('keypoints', None)
        bbox_format = label.pop('bbox_format')
        normalized = label.pop('normalized')
        label['instances'] = Instances(bboxes, segments, bbox_format=bbox_format, normalized=normalized)
        return label

    @staticmethod
    def collate_fn(batch):
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == 'img':
                value = torch.stack(value, 0)
            if k in ('bboxes', 'cls'):
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch['batch_idx'] = list(new_batch['batch_idx'])
        for i in range(len(new_batch['batch_idx'])):
            new_batch['batch_idx'][i] += i
        new_batch['batch_idx'] = torch.cat(new_batch['batch_idx'], 0)
        return new_batch


def load_dataset_cache_file(path):
    import gc
    gc.disable()
    cache = np.load(str(path), allow_pickle=True).item()
    gc.enable()
    return cache


def save_dataset_cache_file(prefix, path, x):
    x['version'] = DATASET_CACHE_VERSION
    if is_dir_writeable(path.parent):
        if path.exists():
            path.unlink()
        np.save(str(path), x)
        path.with_suffix('.cache.npy').rename(path)
        LOGGER.info(f'{prefix}New cache created: {path}')
    else:
        LOGGER.warning(f'{prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable, cache not saved.')
