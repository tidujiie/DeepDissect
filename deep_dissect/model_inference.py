import torch
import tqdm
from mmdet.apis import init_detector, inference_detector
from mmengine.structures import InstanceData
from deep_dissect.utility import get_dataset, load_annotations, parse_image_id_from_path
from PIL import Image
import json
import numpy as np
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)


def compute_predictions_torch_hub(model, dataset):
    """
    Computes predictions for a given dataset using the specified model.

    Parameters:
    model (object): The (pytorch) model used for prediction. It should be compatible with the dataset.
    dataset (object): A dataset object that provides images and targets through its __getitem__ method.

    Returns:
    list: A list of predictions made by the model for all items in the dataset.
    """
    predictions = []
    with torch.no_grad():
        for i in tqdm.tqdm(range(len(dataset))):
            image, target = dataset[i]
            out = model([image.to('cuda')])
            res = out
            predictions.append(res)

    preds = predictions
    return preds


def compute_predictions_mmdet(PATH_CONFIG_FILE, PATH_CHECKPOINT_FILE_BASELINE, CONTENT_PATH, PATH_IMG, device):
    """
    Computes predictions for images located at a specified path using an MMDetection model.

    Parameters:
    PATH_CONFIG_FILE (str): Path to the mmdetection config file
    PATH_CHECKPOINT_FILE_BASELINE (str): Path to the checkpoint file containing the model's state dictionary
    CONTENT_PATH (list): A list of image filenames to perform inference on
    PATH_IMG (str): The base path where the images are stored
    device (str): device that is used for the computation

    Returns:
    torch.Tensor: A tensor containing all bounding boxes predicted by the model across
                  the provided images. Each row in the tensor represents a bounding box.
    """
    model = init_detector(PATH_CONFIG_FILE, PATH_CHECKPOINT_FILE_BASELINE, device=device)
    predictions = []
    with torch.no_grad():
        for i in tqdm.tqdm(CONTENT_PATH):
            out = inference_detector(model, PATH_IMG + i)
            res = out.pred_instances.bboxes
            predictions.append(res)

    preds = torch.cat(predictions, 0)
    return preds


def compute_mmdet_results_unsorted(PATH_CONFIG_FILE, PATH_CHECKPOINT_FILE_BASELINE, PATH_COCO, CONTENT_PATH, PATH_IMG,
                                   device):
    """
    Computes unsorted bounding box regression and classification results for a set of images
    using a specified MMDetection model.

    Parameters:
    PATH_CONFIG_FILE (str): Path to the mmdetection config file
    PATH_CHECKPOINT_FILE_BASELINE (str): Path to the checkpoint file containing the model's state dictionary
    CONTENT_PATH (list): A list of image filenames to perform inference on
    PATH_IMG (str): The base path where the images are stored
    device (str): device that is used for the computation


    Returns:
    tuple: Two tensors and a dictionary, the tensor contain the aggregated unsorted outputs from the bounding box
           regression and classification layers of the model, while the dictionary contains information about the image
           (filename, width, height). The first tensor (`preds`) contains regression results, and the second tensor
           (`cls`) contains classification results.
    """
    model = init_detector(PATH_CONFIG_FILE, PATH_CHECKPOINT_FILE_BASELINE, device=device)
    res_fc_reg = []
    res_fc_cls = []
    image_info = []

    hooks = [
        model.bbox_head.fc_reg.register_forward_hook(
            lambda self, input, output: res_fc_reg.append(output)
        ),
        model.bbox_head.fc_cls.register_forward_hook(
            lambda self, input, output: res_fc_cls.append(output)
        ),
    ]

    dataset = get_dataset(PATH_COCO)
    with torch.no_grad():
        for i in tqdm.tqdm(CONTENT_PATH):
            with Image.open(PATH_IMG + i) as img:
                width, height = img.size
                image_info.append({'name': i, 'width': width, 'height': height})
            out = inference_detector(model, PATH_IMG + i)

    res_fc_reg = res_fc_reg
    res_fc_cls = res_fc_cls

    for hook in hooks:
        hook.remove()

    preds = torch.cat([t.squeeze(0) for t in res_fc_reg], dim=0)
    cls = torch.cat([t.squeeze(0) for t in res_fc_cls], dim=0)
    return preds, cls, image_info