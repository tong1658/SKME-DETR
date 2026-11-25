from torch import nn
import inspect
from pathlib import Path
from typing import Union

from ultralytics.cfg import TASK2DATA, get_cfg, get_save_dir
from ultralytics.nn.tasks import attempt_load_one_weight, guess_model_task, yaml_model_load
from ultralytics.utils import DEFAULT_CFG_DICT, RANK, callbacks, checks, emojis, yaml_load


class Model(nn.Module):

    def __init__(self, model: Union[str, Path], task=None) -> None:
        super().__init__()
        self.callbacks = callbacks.get_default_callbacks()
        self.predictor = None
        self.model = None
        self.trainer = None
        self.ckpt = None
        self.cfg = None
        self.ckpt_path = None
        self.overrides = {}
        self.metrics = None
        self.task = task
        model = str(model).strip()

        suffix = Path(model).suffix
        if suffix in ('.yaml', '.yml'):
            self._new(model, task)
        else:
            self._load(model, task)

    def __call__(self, source=None, stream=False, **kwargs):
        return self.predict(source, stream, **kwargs)

    def _new(self, cfg: str, task=None, model=None, verbose=True):
        cfg_dict = yaml_model_load(cfg)
        self.cfg = cfg
        self.task = task or guess_model_task(cfg_dict)
        self.model = (model or self._smart_load('model'))(cfg_dict, verbose=verbose and RANK == -1)
        self.overrides['model'] = self.cfg
        self.overrides['task'] = self.task

        self.model.args = {**DEFAULT_CFG_DICT, **self.overrides}
        self.model.task = self.task

    def _load(self, weights: str, task=None):
        self.model, self.ckpt = attempt_load_one_weight(weights)
        self.task = self.model.args['task']
        self.overrides = self.model.args = self._reset_ckpt_args(self.model.args)
        self.ckpt_path = self.model.pt_path
        self.overrides['model'] = weights
        self.overrides['task'] = self.task

    def _check_is_pytorch_model(self):
        if not isinstance(self.model, nn.Module):
            raise TypeError('This method requires a PyTorch RT-DETR model.')

    def reset_weights(self):
        self._check_is_pytorch_model()
        for m in self.model.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        for p in self.model.parameters():
            p.requires_grad = True
        return self

    def load(self, weights):
        self._check_is_pytorch_model()
        if isinstance(weights, (str, Path)):
            weights, self.ckpt = attempt_load_one_weight(weights)
        self.model.load(weights)
        return self

    def info(self, detailed=False, verbose=True):
        self._check_is_pytorch_model()
        return self.model.info(detailed=detailed, verbose=verbose)

    def fuse(self):
        self._check_is_pytorch_model()
        self.model.fuse()

    def predict(self, source=None, stream=False, predictor=None, **kwargs):
        if source is None:
            raise ValueError('Pass an image, video, or tensor as the prediction source.')

        custom = {'conf': 0.25, 'save': False}
        args = {**self.overrides, **custom, **kwargs, 'mode': 'predict'}

        if not self.predictor:
            self.predictor = (predictor or self._smart_load('predictor'))(overrides=args, _callbacks=self.callbacks)
            self.predictor.setup_model(model=self.model, verbose=False)
        else:
            self.predictor.args = get_cfg(self.predictor.args, args)
            if 'project' in args or 'name' in args:
                self.predictor.save_dir = get_save_dir(self.predictor.args)
        return self.predictor(source=source, stream=stream)

    def val(self, validator=None, **kwargs):
        custom = {'rect': True}
        args = {**self.overrides, **custom, **kwargs, 'mode': 'val'}

        validator = (validator or self._smart_load('validator'))(args=args, _callbacks=self.callbacks)
        validator(model=self.model)
        self.metrics = validator.metrics
        return validator.metrics

    def train(self, trainer=None, **kwargs):
        self._check_is_pytorch_model()
        overrides = yaml_load(checks.check_yaml(kwargs['cfg'])) if kwargs.get('cfg') else self.overrides
        custom = {'data': TASK2DATA[self.task]}
        args = {**overrides, **custom, **kwargs, 'mode': 'train'}
        if args.get('resume') is True and self.ckpt_path:
            args['resume'] = self.ckpt_path

        self.trainer = (trainer or self._smart_load('trainer'))(overrides=args, _callbacks=self.callbacks)
        if not args.get('resume'):
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            self.model = self.trainer.model
        self.trainer.train()
        if RANK in (-1, 0):
            ckpt = self.trainer.best if self.trainer.best.exists() else self.trainer.last
            self.model, _ = attempt_load_one_weight(ckpt)
            self.overrides = self.model.args
            self.metrics = getattr(self.trainer.validator, 'metrics', None)
        return self.metrics

    def _apply(self, fn):
        self._check_is_pytorch_model()
        self = super()._apply(fn)
        self.predictor = None
        self.overrides['device'] = self.device
        return self

    @property
    def names(self):
        return self.model.names if hasattr(self.model, 'names') else None

    @property
    def device(self):
        return next(self.model.parameters()).device if isinstance(self.model, nn.Module) else None

    @property
    def transforms(self):
        return self.model.transforms if hasattr(self.model, 'transforms') else None

    def add_callback(self, event: str, func):
        self.callbacks[event].append(func)

    def clear_callback(self, event: str):
        self.callbacks[event] = []


    @staticmethod
    def _reset_ckpt_args(args):
        include = {'imgsz', 'data', 'task', 'single_cls'}
        return {k: v for k, v in args.items() if k in include}

    def _smart_load(self, key):
        try:
            return self.task_map[self.task][key]
        except Exception as e:
            name = self.__class__.__name__
            mode = inspect.stack()[1][3]
            raise NotImplementedError(
                emojis(f"WARNING ⚠️ '{name}' model does not support '{mode}' mode for '{self.task}' task yet.")) from e

    @property
    def task_map(self):
        raise NotImplementedError('Please provide task map for your model!')
