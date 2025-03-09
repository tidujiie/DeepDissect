import torch.utils.data
import torchvision

from .compute_ablation import *
from .compute_metrics import *
from .compute_integrated_grouth_truth import *
from .model_inference import *
from .utility import *
from .visualisation import *

from .coco import build

def build_dataset(image_set, args):
    if args.dataset_file == 'coco':
        return build(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
