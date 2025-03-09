import torch
from mmdet.apis import init_detector, inference_detector
from mmengine.structures import InstanceData
from deep_dissect.utility import load_annotations, parse_image_id_from_path


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

    img_meta = det_data_sample.img_metas[0] if hasattr(det_data_sample, 'img_metas') and len(
        det_data_sample.img_metas) > 0 else {}
    gt_instances = InstanceData(bboxes=bboxes_tensor, labels=labels_tensor, metainfo=img_meta)

    det_data_sample.gt_instances = gt_instances

    return det_data_sample


def run_inference_and_integrate_ground_truth(PATH_CONFIG_FILE, PATH_CHECKPOINT_FILE, PATH_ANNOTATIONS, full_paths,
                                             device):
    """
    Initializes a detector model, runs inference on a list of image paths, and integrates the inference results
    with ground truth annotations.

    Parameters:
    - PATH_CONFIG_FILE (str): Path to the configuration file of the model.
    - PATH_CHECKPOINT_FILE (str): Path to the checkpoint file of the model.
    - PATH_ANNOTATIONS (str): Base path to Annotation file where the annotations are stored.
    - full_paths (list): Full paths to the images on which inference is to be performed.
    - device (str): Device for computation 'cuda' or 'cpu'

    Returns:
    - out_updated (list): Inference results integrated with ground truth annotations.
    """

    model = init_detector(PATH_CONFIG_FILE, PATH_CHECKPOINT_FILE, device=device)

    out = inference_detector(model, full_paths)

    annotations = load_annotations(PATH_ANNOTATIONS)

    out_updated = []
    for i in out:
        img_id = parse_image_id_from_path(i.metainfo['img_path'])
        out_updated.append(integrate_ground_truth(i, annotations, img_id))

    return out_updated
