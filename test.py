import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
from prettytable import PrettyTable
from ultralytics import RTDETR
from ultralytics.utils.torch_utils import model_info
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import matplotlib
matplotlib.use('TkAgg')

def get_weight_size(path):
    stats = os.stat(path)
    return f'{stats.st_size / 1024 / 1024:.1f}'

if __name__ == '__main__':
    model_path = r'best.pt'

    model = RTDETR(model_path)
    result = model.val(data='dataset/data.yaml',
                      split='test',
                      imgsz=640,
                      batch=32,
                      save_json=True,
                      project='runs',
                      name='test',
                      )

