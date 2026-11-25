from copy import copy

import numpy as np

from ultralytics.engine.trainer import BaseTrainer
from ultralytics.data import build_dataloader
from ultralytics.utils.plotting import plot_images, plot_labels, plot_results
from ultralytics.utils.torch_utils import torch_distributed_zero_first
from ultralytics.nn.tasks import RTDETRDetectionModel
from ultralytics.utils import RANK, LOGGER, colorstr

from .val import RTDETRDataset, RTDETRValidator


class RTDETRTrainer(BaseTrainer):

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = RTDETRDetectionModel(cfg, nc=self.data['nc'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def build_dataset(self, img_path, mode='val', batch=None):
        return RTDETRDataset(img_path=img_path,
                             imgsz=self.args.imgsz,
                             batch_size=batch,
                             augment=mode == 'train',
                             hyp=self.args,
                             rect=False,
                             cache=self.args.cache or None,
                             fraction=self.args.fraction if mode == 'train' else 1.0,
                             prefix=colorstr(f'{mode}: '),
                             data=self.data)

    def get_validator(self):
        self.loss_names = 'giou_loss', 'cls_loss', 'l1_loss'
        return RTDETRValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        assert mode in ['train', 'val']
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == 'train'
        if getattr(dataset, 'rect', False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == 'train' else self.args.workers * 2
        return build_dataloader(dataset, batch_size, workers, shuffle, rank)

    def preprocess_batch(self, batch):
        batch['img'] = batch['img'].to(self.device, non_blocking=True).float() / 255
        return batch

    def set_model_attributes(self):
        self.model.nc = self.data['nc']
        self.model.names = self.data['names']
        self.model.args = self.args

    def label_loss_items(self, loss_items=None, prefix='train'):
        keys = [f'{prefix}/{x}' for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        return ('\n' + '%11s' *
                (4 + len(self.loss_names))) % ('Epoch', 'GPU_mem', *self.loss_names, 'Instances', 'Size')

    def plot_training_samples(self, batch, ni):
        plot_images(images=batch['img'],
                    batch_idx=batch['batch_idx'],
                    cls=batch['cls'].squeeze(-1),
                    bboxes=batch['bboxes'],
                    paths=batch['im_file'],
                    fname=self.save_dir / f'train_batch{ni}.jpg',
                    on_plot=self.on_plot)

    def plot_metrics(self):
        plot_results(file=self.csv, on_plot=self.on_plot)

    def plot_training_labels(self):
        boxes = np.concatenate([lb['bboxes'] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb['cls'] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.reshape(-1), names=self.data['names'], save_dir=self.save_dir, on_plot=self.on_plot)
