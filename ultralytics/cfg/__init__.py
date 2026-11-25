from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Union

from ultralytics.utils import (DEFAULT_CFG_DICT, LOGGER, RANK, ROOT, RUNS_DIR,
                               IterableSimpleNamespace, colorstr, yaml_load)

TASK2DATA = {'detect': str(ROOT.parent / 'dataset/data.yaml')}

CFG_FLOAT_KEYS = ('warmup_epochs', 'degrees', 'shear')
CFG_FRACTION_KEYS = ('lr0', 'lrf', 'momentum', 'weight_decay', 'warmup_momentum',
                     'warmup_bias_lr', 'hsv_h', 'hsv_s', 'hsv_v', 'translate', 'scale',
                     'perspective', 'flipud', 'fliplr', 'mosaic', 'mixup', 'copy_paste',
                     'conf', 'fraction')
CFG_INT_KEYS = ('epochs', 'patience', 'batch', 'workers', 'seed', 'close_mosaic',
                'max_det', 'vid_stride', 'line_width', 'nbs', 'save_period')
CFG_BOOL_KEYS = ('save', 'exist_ok', 'verbose', 'deterministic', 'single_cls', 'rect',
                 'cos_lr', 'val', 'save_json', 'half', 'plots', 'show', 'save_txt',
                 'save_conf', 'save_crop', 'show_labels', 'show_conf', 'visualize',
                 'augment', 'boxes', 'amp')


def cfg2dict(cfg):
    if isinstance(cfg, (str, Path)):
        cfg = yaml_load(cfg)
    elif isinstance(cfg, SimpleNamespace):
        cfg = vars(cfg)
    return cfg


def get_cfg(cfg: Union[str, Path, Dict, SimpleNamespace] = DEFAULT_CFG_DICT, overrides: Dict = None):
    cfg = cfg2dict(cfg)

    if overrides:
        overrides = cfg2dict(overrides)
        if 'save_dir' not in cfg:
            overrides = {k: v for k, v in overrides.items() if k != 'save_dir'}
        check_dict_alignment(cfg, overrides)
        cfg = {**cfg, **overrides}

    for k in 'project', 'name':
        if k in cfg and isinstance(cfg[k], (int, float)):
            cfg[k] = str(cfg[k])
    if cfg.get('name') == 'model':
        cfg['name'] = cfg.get('model', '').split('.')[0]
        LOGGER.warning(f"WARNING ⚠️ 'name=model' automatically updated to 'name={cfg['name']}'.")

    for k, v in cfg.items():
        if v is not None:
            if k in CFG_FLOAT_KEYS and not isinstance(v, (int, float)):
                raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. "
                                f"Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')")
            elif k in CFG_FRACTION_KEYS:
                if not isinstance(v, (int, float)):
                    raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. "
                                    f"Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')")
                if not (0.0 <= v <= 1.0):
                    raise ValueError(f"'{k}={v}' is an invalid value. "
                                     f"Valid '{k}' values are between 0.0 and 1.0.")
            elif k in CFG_INT_KEYS and not isinstance(v, int):
                raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. "
                                f"'{k}' must be an int (i.e. '{k}=8')")
            elif k in CFG_BOOL_KEYS and not isinstance(v, bool):
                raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. "
                                f"'{k}' must be a bool (i.e. '{k}=True' or '{k}=False')")

    return IterableSimpleNamespace(**cfg)


def get_save_dir(args, name=None):

    if getattr(args, 'save_dir', None):
        save_dir = args.save_dir
    else:
        from ultralytics.utils.files import increment_path

        project = args.project or RUNS_DIR / args.task
        name = name or args.name or f'{args.mode}'
        save_dir = increment_path(Path(project) / name, exist_ok=args.exist_ok if RANK in (-1, 0) else True)

    return Path(save_dir)


def check_dict_alignment(base: Dict, custom: Dict, e=None):
    base_keys, custom_keys = (set(x.keys()) for x in (base, custom))
    mismatched = [k for k in custom_keys if k not in base_keys]
    if mismatched:
        from difflib import get_close_matches

        string = ''
        for x in mismatched:
            matches = get_close_matches(x, base_keys)
            matches = [f'{k}={base[k]}' if base.get(k) is not None else k for k in matches]
            match_str = f'Similar arguments are i.e. {matches}.' if matches else ''
            string += f"'{colorstr('red', 'bold', x)}' is not a valid YOLO argument. {match_str}\n"
        raise SyntaxError(string) from e
