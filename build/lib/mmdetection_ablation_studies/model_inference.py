import torch
import tqdm
from mmdet.apis import init_detector, inference_detector
from .data_processing import *
from mmengine.structures import InstanceData
import json

def compute_predictions_torch_hub(model, dataset):
    """
    Computes predictions for a given dataset using the specified model.

    Parameters:
    model: The model (checkpoint) used for prediction. It should be compatible with the dataset.
    dataset: A dataset object that provides images and targets through its __getitem__ method.

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
    PATH_CONFIG_FILE: Path to the mmdetection config file 
    PATH_CHECKPOINT_FILE_BASELINE: Path to the checkpoint file containing the model's state dictionary
    CONTENT_PATH: A list of image filenames to perform inference on
    PATH_IMG: The base path where the images are stored
    device: device that is used for the computation

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

def compute_mmdet_results_unsorted(PATH_CONFIG_FILE, PATH_CHECKPOINT_FILE_BASELINE, PATH_COCO, CONTENT_PATH, PATH_IMG, device):
    """
    Computes unsorted bounding box regression and classification results for a set of images
    using a specified MMDetection model.

    Parameters:
    PATH_CONFIG_FILE: Path to the mmdetection config file 
    PATH_CHECKPOINT_FILE_BASELINE: Path to the checkpoint file containing the model's state dictionary
    CONTENT_PATH: A list of image filenames to perform inference on
    PATH_IMG: The base path where the images are stored
    device: device that is used for the computation


    Returns:
    tuple: Two tensors containing the aggregated unsorted outputs from the bounding box regression
           and classification layers of the model. The first tensor (`preds`) contains regression
           results, and the second tensor (`cls`) contains classification results.
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

def load_annotations(annotation_path):
    with open(annotation_path) as file:
        annotations = json.load(file)
    return annotations

def parse_image_id_from_path(img_path):
    """
    Parse the image ID from an image path

    Parameters:
    - img_path: The full path to the image file.

    Returns:
    - image_id: The image ID as an integer.
    """
    filename = img_path.split('/')[-1]

    image_id_str = filename.lstrip('0').rstrip('.jpg')

    image_id = int(image_id_str)

    return image_id

def integrate_ground_truth(det_data_sample, annotations, img_id):
    """
    Integrate ground truth bounding boxes and labels into a detection data sample based on image ID.

    This function processes ground truth annotations for a specific image, identified by its image ID, and integrates 
    these annotations into a provided detection data sample object. It reformats each ground truth bounding box from the 
    format specified in the annotations ([x_min, y_min, width, height]) to a more commonly used format ([x_min, y_min, x_max, y_max]), 
    and converts the list of bounding boxes and labels into PyTorch tensors. These tensors are then assigned to the detection 
    data sample, enriching it with ground truth information necessary for evaluation or training purposes.

    Parameters:
    det_data_sample (object): An object representing a detection data sample that will be updated with ground truth information.
    annotations (dict): A dictionary containing all annotations, where annotations related to the specified image ID will be extracted.
    img_id (int): The identifier for the image whose annotations are to be integrated into the detection data sample.

    Returns:
    object: The detection data sample object updated with ground truth bounding boxes and labels for the specified image.
    """
    annotations_for_image = [ann for ann in annotations['annotations'] if ann['image_id'] == img_id]
    
    bboxes = []
    labels = []
    for ann in annotations_for_image:
        bbox = ann['bbox']
        x1, y1, w, h = bbox
        x2 = x1 + w
        y2 = y1 + h
        bboxes.append([x1, y1, x2, y2])
        labels.append(ann['category_id'])
    
    bboxes_tensor = torch.tensor(bboxes, dtype=torch.float).to('cuda:0')
    labels_tensor = torch.tensor(labels, dtype=torch.int64).to('cuda:0')
    
    img_meta = det_data_sample.img_metas[0] if hasattr(det_data_sample, 'img_metas') and len(det_data_sample.img_metas) > 0 else {}
    gt_instances = InstanceData(bboxes=bboxes_tensor, labels=labels_tensor, metainfo=img_meta)

    det_data_sample.gt_instances = gt_instances
    
    return det_data_sample

def run_inference_and_integrate_ground_truth(PATH_CONFIG_FILE, PATH_CHECKPOINT_FILE, PATH_ANNOTATIONS, full_paths, device):
    """
    Initializes a detector model, runs inference on a list of image paths, and integrates the inference results
    with ground truth annotations.

    Parameters:
    - PATH_CONFIG_FILE: str, path to the configuration file of the model.
    - PATH_CHECKPOINT_FILE: str, path to the checkpoint file of the model.
    - PATH_ANNOTATIONS: str, base path to Annotation file where the annotations are stored.
    - full_paths: list, full paths to the images on which inference is to be performed.
    - device: str, device for computation 'cuda' or 'cpu'

    Returns:
    - out_updated: list, inference results integrated with ground truth annotations.
    """

    model = init_detector(PATH_CONFIG_FILE, PATH_CHECKPOINT_FILE, device=device)
    
    out = inference_detector(model, full_paths)
    
    annotations = load_annotations(PATH_ANNOTATIONS)
    
    out_updated = []
    for i in out:
        img_id = parse_image_id_from_path(i.metainfo['img_path'])
        out_updated.append(integrate_ground_truth(i, annotations, img_id))
    
    return out_updated