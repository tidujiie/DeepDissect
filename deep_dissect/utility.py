import torch
from datasets import build_dataset
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

def get_model_state_dict_and_keys(
        PATH_CHECKPOINT_FILE):
    """
    Retrieve the state dictionary and keys from a checkpoint file.
    This is useful for identifying the naming of the components of a checkpoint file.

    Parameters:
    - PATH_CHECKPOINT_FILE (str): Path to the model's checkpoint file.

    Returns:
    - state_dict (dict): The full state dictionary containing all the model's parameters.
    - keys (list/str): A list of keys with the naming of the components of a checkpoint file.
    """
    model = torch.load(PATH_CHECKPOINT_FILE)
    state_dict = model['state_dict']
    
    keys = [key for key in state_dict.keys() if key.endswith('')]
    
    return state_dict, keys

def normalize_bboxes(bboxes, size):
    """
    Normalizes bounding box coordinates based on the image size, converting 
    them to a scale of 0 to 1. This is useful for preparing bounding boxes for 
    models that expect inputs in a normalized format.

    Parameters:
    bboxes (torch.Tensor): A tensor of shape [N, 4], where N is the number of bounding boxes,
                           and each bounding box is defined by its corner coordinates
                           (xmin, ymin, xmax, ymax).
    size (tuple): A tuple (img_w, img_h) representing the width and height of the image.
                  These dimensions are used to normalize the bounding box coordinates.

    Returns:
    torch.Tensor: A tensor of normalized bounding boxes with the same shape as the input.
                  Each bounding box is now represented by coordinates in the range [0, 1].
    """
    img_w, img_h = size
    normalized_bboxes = bboxes / torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return normalized_bboxes

def box_xyxy_to_cxcywh(x):
    """
    Converts bounding boxes from corner (xmin, ymin, xmax, ymax) format to center (cx, cy, w, h) format.

    Parameters:
    x (torch.Tensor): a tensor of shape [N, 4] where N is the number of boxes, and each box is defined
                      by (xmin, ymin, xmax, ymax).

    Returns:
    torch.Tensor: the converted bounding boxes in (cx, cy, w, h) format.
    """
    top_left_x, top_left_y, bottom_right_x, bottom_right_y = x.unbind(1)
    w = bottom_right_x - top_left_x
    h = bottom_right_y - top_left_y
    x_c = top_left_x + 0.5 * w
    y_c = top_left_y + 0.5 * h
    return torch.stack([x_c, y_c, w, h], dim=1)

def box_cxcywh_to_xyxy(x):
    """
    Converts bounding boxes from center (cx, cy, w, h) format to corner (xmin, ymin, xmax, ymax) format.

    Parameters:
    x (torch.Tensor): a tensor of shape [N, 4] where N is the number of boxes, and each box is defined
                      by (cx, cy, w, h).

    Returns:
    torch.Tensor: the converted bounding boxes in (xmin, ymin, xmax, ymax) format.
    """
    cx, cy, w, h = x.unbind(1)
    x_min = cx - 0.5 * w
    y_min = cy - 0.5 * h
    x_max = cx + 0.5 * w
    y_max = cy + 0.5 * h
    return torch.stack([x_min, y_min, x_max, y_max], dim=1)

def get_image_size(image_path):
    """
    Gets image dimensions using PIL for tasks like resizing and cropping.

    Parameters:
    image_path (str): The file path to the image for which dimensions are to be retrieved.

    Returns:
    tuple: A tuple containing two elements: (width, height), representing the width and 
           height of the image in pixels.
    """
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height

def get_dataset(coco_path):
    """
    Loads a COCO-format dataset for object detection models using a specified path.

    Parameters:
    coco_path (str): The file path to the COCO dataset. This path should contain the 
                     dataset files structured in the COCO format, including annotations 
                     and images.

    Returns:
    Dataset (object): An object representing the loaded dataset, prepared for use with models
             trained on COCO-formatted data.
    """
    class DummyArgs:
        pass
    args = DummyArgs()
    args.dataset_file = "coco"
    args.coco_path = coco_path
    args.masks = False
    dataset = build_dataset(image_set='val', args=args)
    return dataset

def load_annotations(annotation_path):
    """
    Loads annotations from a .json file to a dictionary

    Parameters:
        annotation_path (str): The file path a dataset file in .json format

    Returns:
        annotations (dict): A dictionary containing the annotations without any further processing.
    """
    with open(annotation_path) as file:
        annotations = json.load(file)
    return annotations

def parse_image_id_from_path(img_path):
    """
    Parse the image ID from an image path

    Parameters:
    - img_path (str): The full path to the image file.

    Returns:
    - image_id (int): The image ID as an integer.
    """
    filename = img_path.split('/')[-1]

    image_id_str = filename.lstrip('0').rstrip('.jpg')

    image_id = int(image_id_str)

    return image_id