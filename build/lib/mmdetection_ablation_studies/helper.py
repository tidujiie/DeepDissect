import torch
import json
from mmengine.structures import InstanceData

def find_zero_rows(model_weights):
    """
    Identifies rows in a tensor that are entirely zero.

    Parameters:
    model_weights (torch.Tensor): A tensor representing the weights of a model, 
                                  where each row corresponds to a different set of weights.

    Returns:
    list: A list of indices representing the rows in `model_weights` that are completely zero.
    """
    zero_rows = []

    for i, row in enumerate(model_weights):
        if torch.all(row == 0):
            zero_rows.append(i)

    return zero_rows

def load_annotations(annotation_path):
    """
    Load annotations from a JSON file.

    This function opens a JSON file specified by the 'annotation_path' parameter and loads its contents into a Python dictionary. 
    JSON files are commonly used for storing annotations in various machine learning tasks, including object detection and image classification. 
    The annotations may include information such as class labels, bounding boxes, and other metadata related to the dataset.

    Parameters:
    annotation_path (str): The file path to the JSON file containing annotations.

    Returns:
    dict: A dictionary containing the loaded annotations from the JSON file.
    """
    with open(annotation_path) as file:
        annotations = json.load(file)
    return annotations

def parse_image_id_from_path(img_path):
    """
    Extract the image ID from a file path.

    This function parses the file path of an image to extract its ID. It assumes that the image file name consists
    of a series of leading zeros followed by the image ID and then the '.jpg' file extension. The function splits the
    path to isolate the file name, removes leading zeros and the file extension to isolate the image ID, and then
    converts this ID from a string to an integer.

    Parameters:
    img_path (str): The full path to the image file, including directories and file extension.

    Returns:
    int: The numerical image ID extracted from the file name.
    """
    filename = img_path.split('/')[-1]

    image_id_str = filename.lstrip('0').rstrip('.jpg')

    image_id = int(image_id_str)

    return image_id