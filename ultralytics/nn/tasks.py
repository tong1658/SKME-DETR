import inspect
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from ultralytics.nn.extra_modules import (
    CA_DBBC3, CGNet, CSP_EAM, DP_DHTB, SCGM, SDFM,
)
from ultralytics.nn.modules import Concat, Conv, RTDETRDecoder
from ultralytics.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, yaml_load
from ultralytics.utils.checks import check_suffix, check_yaml
from ultralytics.utils.torch_utils import (
    fuse_conv_and_bn, initialize_weights, intersect_dicts, make_divisible, model_info, time_sync,
)


DEFAULT_MODEL = Path(__file__).resolve().parents[2] / 'gaijin.yaml'
MODEL_LAYERS = {layer.__name__: layer for layer in (
    Conv, SCGM, CSP_EAM, DP_DHTB,
    Concat, CA_DBBC3, CGNet, SDFM, RTDETRDecoder,
)}
MODEL_LAYERS['nn.Upsample'] = nn.Upsample

MODEL_LAYERS.update({
    'StemSafeGate': SCGM,
    'CSP_MutilScaleEdgeInformationEnhance': CSP_EAM,
    'TransformerEncoderLayer_DHSA': DP_DHTB,
    'DBBC3': CA_DBBC3,
    'ContextGuidedBlock_Down': CGNet,
})


class BaseModel(nn.Module):

    def forward(self, x, *args, **kwargs):
        if isinstance(x, dict):
            return self.loss(x, *args, **kwargs)
        return self.predict(x, *args, **kwargs)

    def load(self, weights, verbose=True):
        model = weights['model'] if isinstance(weights, dict) else weights
        state = intersect_dicts(model.float().state_dict(), self.state_dict())
        self.load_state_dict(state, strict=False)
        if verbose:
            LOGGER.info(f'Transferred {len(state)}/{len(self.state_dict())} items from pretrained weights')

    def info(self, detailed=False, verbose=True, imgsz=640):
        return model_info(self, detailed=detailed, verbose=verbose, imgsz=imgsz)

    def is_fused(self, thresh=10):
        norms = tuple(cls for name, cls in vars(nn).items() if 'Norm' in name)
        return sum(isinstance(m, norms) for m in self.modules()) < thresh

    def fuse(self, verbose=True):
        for layer in self.model.modules():
            if isinstance(layer, Conv) and hasattr(layer, 'bn'):
                layer.conv = fuse_conv_and_bn(layer.conv, layer.bn)
                del layer.bn
                layer.forward = layer.forward_fuse
            if hasattr(layer, 'switch_to_deploy'):
                layer.switch_to_deploy()
        self.info(verbose=verbose)
        return self

    def _profile_one_layer(self, layer, x, timings):
        start = time_sync()
        for _ in range(10):
            layer(x)
        timings.append((time_sync() - start) * 100)
        LOGGER.info(f'{timings[-1]:10.2f} ms  {layer.np:10d} parameters  {layer.type}')


class DetectionModel(BaseModel):

    def __init__(self, cfg=DEFAULT_MODEL, ch=3, nc=None, verbose=True):
        super().__init__()
        self.yaml = deepcopy(cfg) if isinstance(cfg, dict) else yaml_model_load(cfg)
        self.yaml['ch'] = self.yaml.get('ch', ch)
        if nc is not None and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc
        self.nc = self.yaml['nc']
        self.model, self.save = parse_model(self.yaml, self.yaml['ch'], verbose)
        self.names = {i: str(i) for i in range(self.nc)}
        self.inplace = self.yaml.get('inplace', True)
        self.stride = torch.tensor([32.])
        with torch.no_grad():
            self.predict(torch.zeros(2, self.yaml['ch'], 640, 640))
        initialize_weights(self)
        if verbose:
            self.info()

    def predict(self, x, profile=False, visualize=False, batch=None, augment=False):
        outputs, timings = [], []
        for layer in self.model[:-1]:
            if isinstance(layer.f, list):
                x = [x if source == -1 else outputs[source] for source in layer.f]
            elif layer.f != -1:
                x = outputs[layer.f]
            if profile:
                self._profile_one_layer(layer, x, timings)
            x = layer(x)
            outputs.append(x if layer.i in self.save else None)
            if visualize:
                from ultralytics.utils.plotting import feature_visualization
                feature_visualization(x, layer.type, layer.i, save_dir=visualize)
        head = self.model[-1]
        return head([outputs[source] for source in head.f], batch)


class RTDETRDetectionModel(DetectionModel):

    def init_criterion(self):
        from ultralytics.models.utils.loss import RTDETRDetectionLoss
        return RTDETRDetectionLoss(nc=self.nc, use_vfl=True)

    def loss(self, batch, preds=None):
        if not hasattr(self, 'criterion'):
            self.criterion = self.init_criterion()
        img = batch['img']
        batch_idx = batch['batch_idx'].to(device=img.device, dtype=torch.long).view(-1)
        targets = {
            'cls': batch['cls'].to(device=img.device, dtype=torch.long).view(-1),
            'bboxes': batch['bboxes'].to(img.device),
            'batch_idx': batch_idx,
            'gt_groups': [(batch_idx == i).sum().item() for i in range(len(img))],
        }
        if preds is None:
            preds = self.predict(img, batch=targets)
        dec_boxes, dec_scores, enc_boxes, enc_scores, dn_meta = preds if self.training else preds[1]
        dn_boxes = dn_scores = None
        if dn_meta is not None:
            dn_boxes, dec_boxes = torch.split(dec_boxes, dn_meta['dn_num_split'], dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, dn_meta['dn_num_split'], dim=2)
        boxes = torch.cat([enc_boxes.unsqueeze(0), dec_boxes])
        scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])
        loss = self.criterion((boxes, scores), targets, dn_bboxes=dn_boxes, dn_scores=dn_scores, dn_meta=dn_meta)
        items = torch.stack([loss[k].detach() for k in ('loss_giou', 'loss_class', 'loss_bbox')])
        return sum(loss.values()), items


def parse_model(d, ch, verbose=True):
    nc = d['nc']
    scales = d.get('scales')
    if scales:
        scale = d.get('scale') or next(iter(scales))
        depth, width, max_channels = scales[scale]
    else:
        depth, width, max_channels = d.get('depth_multiple', 1.), d.get('width_multiple', 1.), float('inf')
    channels = [ch]
    layers, save = [], set()
    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>14}{'n':>3}{'params':>10}  {'module':<40}arguments")
    for i, (source, repeats, name, yaml_args) in enumerate(d['backbone'] + d['head']):
        if name not in MODEL_LAYERS:
            raise ValueError(f'Unknown layer {name!r} at index {i} in the RT-DETR model configuration')
        layer_type = MODEL_LAYERS[name]
        args = [{'nc': nc, 'None': None}.get(arg, arg) if isinstance(arg, str) else arg for arg in yaml_args]
        repeats = max(round(repeats * depth), 1) if repeats > 1 else repeats
        count = repeats
        if layer_type in (Conv, CA_DBBC3, CSP_EAM):
            out_ch = args[0]
            if out_ch != nc:
                out_ch = make_divisible(min(out_ch, max_channels) * width, 8)
            args = [channels[source], out_ch, *args[1:]]
            if layer_type in (CA_DBBC3, CSP_EAM):
                args.insert(2, repeats)
                count = 1
        elif layer_type is DP_DHTB:
            out_ch = channels[source]
            args = [out_ch, *args]
        elif layer_type is CGNet:
            out_ch = 2 * channels[source]
            args = [channels[source], out_ch, *args]
        elif layer_type is Concat:
            out_ch = sum(channels[j] for j in source)
        elif layer_type is SDFM:
            out_ch = channels[source[1]]
            args = [out_ch, *args]
        elif layer_type is RTDETRDecoder:
            args.insert(1, [channels[j] for j in source])
            out_ch = nc + 4
        else:
            out_ch = channels[source]

        layer = nn.Sequential(*(layer_type(*args) for _ in range(count))) if count > 1 else layer_type(*args)
        layer.i, layer.f = i, source
        layer.type = f'{layer_type.__module__}.{layer_type.__name__}'
        layer.np = sum(param.numel() for param in layer.parameters())
        if verbose:
            LOGGER.info(f'{i:>3}{str(source):>14}{repeats:>3}{layer.np:>10}  {name:<40}{args}')
        save.update(j % i for j in ([source] if isinstance(source, int) else source) if j != -1)
        layers.append(layer)
        if i == 0:
            channels.clear()
        channels.append(out_ch)
    return nn.Sequential(*layers), sorted(save)


def yaml_model_load(path):
    config = yaml_load(check_yaml(str(path)))
    config['yaml_file'] = str(path)
    return config


def guess_model_task(model):
    return 'detect'


def torch_safe_load(weight):
    check_suffix(file=weight, suffix='.pt')
    kwargs = {'weights_only': False} if 'weights_only' in inspect.signature(torch.load).parameters else {}
    return torch.load(weight, map_location='cpu', **kwargs), str(weight)


def attempt_load_one_weight(weight, device=None, inplace=True, fuse=False):
    ckpt, path = torch_safe_load(weight)
    model = (ckpt.get('ema') or ckpt['model']).to(device).float().eval()
    args = {**DEFAULT_CFG_DICT, **ckpt.get('train_args', {})}
    model.args = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}
    model.pt_path, model.task = path, 'detect'
    for m in model.modules():
        if isinstance(m, (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU)):
            m.inplace = inplace
        elif isinstance(m, nn.Upsample) and not hasattr(m, 'recompute_scale_factor'):
            m.recompute_scale_factor = None
    if fuse:
        model.fuse()
    return model, ckpt


def attempt_load_weights(weights, device=None, inplace=True, fuse=False):
    if isinstance(weights, (list, tuple)):
        if len(weights) != 1:
            raise ValueError('Expected one RT-DETR checkpoint')
        weights = weights[0]
    return attempt_load_one_weight(weights, device, inplace, fuse)[0]
