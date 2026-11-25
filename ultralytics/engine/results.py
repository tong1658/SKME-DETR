from copy import deepcopy
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from ultralytics.utils import SimpleClass, ops
from ultralytics.utils.plotting import Annotator, colors, save_one_box


class BaseTensor(SimpleClass):

    def __init__(self, data, orig_shape) -> None:
        assert isinstance(data, (torch.Tensor, np.ndarray))
        self.data = data
        self.orig_shape = orig_shape

    @property
    def shape(self):
        return self.data.shape

    def cpu(self):
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.cpu(), self.orig_shape)

    def numpy(self):
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.numpy(), self.orig_shape)

    def cuda(self):
        return self.__class__(torch.as_tensor(self.data).cuda(), self.orig_shape)

    def to(self, *args, **kwargs):
        return self.__class__(torch.as_tensor(self.data).to(*args, **kwargs), self.orig_shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.__class__(self.data[idx], self.orig_shape)


class Results(SimpleClass):

    def __init__(self, orig_img, path, names, boxes=None) -> None:
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]
        self.boxes = Boxes(boxes, self.orig_shape) if boxes is not None else None
        self.speed = {'preprocess': None, 'inference': None, 'postprocess': None}
        self.names = names
        self.path = path
        self.save_dir = None
        self._keys = ('boxes',)

    def __getitem__(self, idx):
        return self._apply('__getitem__', idx)

    def __len__(self):
        return len(self.boxes) if self.boxes is not None else 0

    def update(self, boxes):
        ops.clip_boxes(boxes, self.orig_shape)
        self.boxes = Boxes(boxes, self.orig_shape)

    def _apply(self, fn, *args, **kwargs):
        r = self.new()
        for k in self._keys:
            v = getattr(self, k)
            if v is not None:
                setattr(r, k, getattr(v, fn)(*args, **kwargs))
        return r

    def cpu(self):
        return self._apply('cpu')

    def numpy(self):
        return self._apply('numpy')

    def cuda(self):
        return self._apply('cuda')

    def to(self, *args, **kwargs):
        return self._apply('to', *args, **kwargs)

    def new(self):
        return Results(orig_img=self.orig_img, path=self.path, names=self.names)

    def plot(self, conf=True, line_width=None, font_size=None, font='Arial.ttf', pil=False,
             img=None, labels=True, boxes=True):
        if img is None and isinstance(self.orig_img, torch.Tensor):
            img = (self.orig_img[0].detach().permute(1, 2, 0).contiguous() * 255).to(torch.uint8).cpu().numpy()

        names = self.names
        pred_boxes, show_boxes = self.boxes, boxes
        annotator = Annotator(deepcopy(self.orig_img if img is None else img),
                              line_width, font_size, font, pil, example=names)

        if pred_boxes and show_boxes:
            for d in reversed(pred_boxes):
                c, score, id = int(d.cls), float(d.conf), None if d.id is None else int(d.id.item())
                name = ('' if id is None else f'id:{id} ') + names[c]
                label = (f'{name} {score:.2f}' if conf else name) if labels else None
                annotator.box_label(d.xyxy.squeeze(), label, color=colors(c, True))

        return annotator.result()

    def verbose(self):
        log_string = ''
        boxes = self.boxes
        if len(self) == 0:
            return '(no detections), '
        if boxes:
            for c in boxes.cls.unique():
                n = (boxes.cls == c).sum()
                log_string += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "
        return log_string

    def save_txt(self, txt_file, save_conf=False):
        boxes = self.boxes
        texts = []
        if boxes:
            for d in boxes:
                c, conf, id = int(d.cls), float(d.conf), None if d.id is None else int(d.id.item())
                line = (c, *d.xywhn.view(-1))
                line += (conf, ) * save_conf + (() if id is None else (id, ))
                texts.append(('%g ' * len(line)).rstrip() % line)

        if texts:
            Path(txt_file).parent.mkdir(parents=True, exist_ok=True)
            with open(txt_file, 'a') as f:
                f.writelines(text + '\n' for text in texts)

    def save_crop(self, save_dir, file_name=Path('im.jpg')):
        for d in self.boxes:
            save_one_box(d.xyxy,
                         self.orig_img.copy(),
                         file=Path(save_dir) / self.names[int(d.cls)] / f'{Path(file_name).stem}.jpg',
                         BGR=True)

    def tojson(self, normalize=False):
        import json

        results = []
        data = self.boxes.data.cpu().tolist()
        h, w = self.orig_shape if normalize else (1, 1)
        for i, row in enumerate(data):
            box = {'x1': row[0] / w, 'y1': row[1] / h, 'x2': row[2] / w, 'y2': row[3] / h}
            conf = row[-2]
            class_id = int(row[-1])
            name = self.names[class_id]
            result = {'name': name, 'class': class_id, 'confidence': conf, 'box': box}
            if self.boxes.is_track:
                result['track_id'] = int(row[-3])
            results.append(result)

        return json.dumps(results, indent=2)


class Boxes(BaseTensor):

    def __init__(self, boxes, orig_shape) -> None:
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        n = boxes.shape[-1]
        assert n in (6, 7), f'expected `n` in [6, 7], but got {n}'
        super().__init__(boxes, orig_shape)
        self.is_track = n == 7
        self.orig_shape = orig_shape

    @property
    def xyxy(self):
        return self.data[:, :4]

    @property
    def conf(self):
        return self.data[:, -2]

    @property
    def cls(self):
        return self.data[:, -1]

    @property
    def id(self):
        return self.data[:, -3] if self.is_track else None

    @property
    @lru_cache(maxsize=2)
    def xywh(self):
        return ops.xyxy2xywh(self.xyxy)

    @property
    @lru_cache(maxsize=2)
    def xyxyn(self):
        xyxy = self.xyxy.clone() if isinstance(self.xyxy, torch.Tensor) else np.copy(self.xyxy)
        xyxy[..., [0, 2]] /= self.orig_shape[1]
        xyxy[..., [1, 3]] /= self.orig_shape[0]
        return xyxy

    @property
    @lru_cache(maxsize=2)
    def xywhn(self):
        xywh = ops.xyxy2xywh(self.xyxy)
        xywh[..., [0, 2]] /= self.orig_shape[1]
        xywh[..., [1, 3]] /= self.orig_shape[0]
        return xywh
