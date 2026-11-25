import torch
from torch import nn


def check_class_names(names):
    if isinstance(names, list):
        names = dict(enumerate(names))
    names = {int(index): str(label) for index, label in names.items()}
    if not names or sorted(names) != list(range(len(names))):
        raise ValueError('Class indices must run consecutively from 0.')
    return names


class AutoBackend(nn.Module):

    pt = True

    @torch.no_grad()
    def __init__(self, weights, device=torch.device('cpu'),
                 fp16=False, fuse=True, verbose=True):
        super().__init__()
        self.device = torch.device(device)
        self.fp16 = fp16 and self.device.type == 'cuda'

        if isinstance(weights, nn.Module):
            model = weights.to(self.device)
            if fuse:
                model = model.fuse(verbose=verbose)
        else:
            from ultralytics.nn.tasks import attempt_load_weights
            model = attempt_load_weights(weights, device=self.device, fuse=fuse)

        self.model = (model.half() if self.fp16 else model.float()).eval()
        self.stride = max(32, int(model.stride.max()))
        self.names = check_class_names(model.names)

    def forward(self, im, augment=False, visualize=False):
        if self.fp16:
            im = im.half()
        return self.model(im, augment=augment, visualize=visualize)

    @torch.no_grad()
    def warmup(self, imgsz):
        if self.device.type == 'cuda':
            dtype = torch.float16 if self.fp16 else torch.float32
            self(torch.zeros(imgsz, dtype=dtype, device=self.device))
