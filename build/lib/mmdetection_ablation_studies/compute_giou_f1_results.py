from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou
from sklearn.metrics import f1_score
import pickle
import json
import os

CLASSES_MMDET = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush')

CLASSES_DETR = [
    'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

class_mapping = {}
for i, class_name in enumerate(CLASSES_MMDET):
    if class_name in CLASSES_DETR and class_name != 'N/A':
        detr_index = CLASSES_DETR.index(class_name)
        class_mapping[i] = detr_index

def calculate_average_giou_overall_and_save_results(input_path, output_path, iterations, start_idx=0):
    """
    This function processes a series of pickle files, extracts ground-truth and predicted bounding boxes, and calculates the average GIoU (Generalized Intersection over Union) for the entire dataset (across all classes).

    Parameters:
    - input_path (str): The path where the pickle files are stored. The filename should include the index.
    Example: "ablation_studies/decoder_cross_attention_30/integrated_ground_truth_decoder_cross_attention_30_"
    - output_path (str): The path where the result files (JSON) should be stored.
    Example: "ablation_studies/decoder_cross_attention_30/"
    - iterations (int): The number of files to process, starting from the start index.
    - start_idx (int): The starting index for the files. Default is 0.

    Returns:
    - The function saves the results as JSON files in the output path and prints the calculated values to the console.
    """
    
    for file_idx in range(start_idx, start_idx + iterations):
        file_name = f'{input_path}{file_idx}.pkl'
        
        if not os.path.exists(file_name):
            print(f"File {file_name} does not exist. Skip...")
            continue

        with open(file_name, 'rb') as file:
            results_gt_pred = pickle.load(file)

        total_giou = 0
        total_pairs = 0

        for result in results_gt_pred:
            gt_bboxes = result.gt_instances.bboxes
            pred_bboxes = result.pred_instances.bboxes

            if gt_bboxes.numel() == 0 or pred_bboxes.numel() == 0:
                continue

            giou_matrix = generalized_box_iou(gt_bboxes, pred_bboxes)

            cost_matrix = 1 - giou_matrix.cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            num_pairs = len(row_ind)
            total_pairs += num_pairs
            for gt_idx, pred_idx in zip(row_ind, col_ind):
                giou_value = giou_matrix[gt_idx, pred_idx].item()
                total_giou += giou_value

        average_giou = total_giou / total_pairs if total_pairs > 0 else 0

        output_data = {
            "average_giou": average_giou
        }

        output_file_name = f'bbox_regression_gt_pred_average_giou_overall_result_{file_idx}.json'
        with open(output_path + output_file_name, 'w') as json_file:
            json.dump(output_data, json_file)

        print(f"Processing iteration: {file_idx+1}/{iterations}")



def calculate_average_giou_per_class_and_save_results(input_path, output_path, iterations, start_idx=0):
    """
    This function processes a series of Pickle files, extracts ground truth and predicted bounding boxes,
    and calculates the average GIoU (Generalized Intersection over Union) per class.
    
    Parameters:
    - input_path (str): The path where the Pickle files are stored. The filename should include the index.
      Example: "ablation_studies/decoder_cross_attention_30/integrated_ground_truth_decoder_cross_attention_30_"
    - output_path (str): Path to save the json results.
    - iterations (int): The number of files to process, starting from the start index.
    - start_idx (int): The starting index for the files. Default is 0.
    
    
    Returns:
    - The function saves the results as JSON files in the specified directory.
    """

    for file_idx in range(start_idx, start_idx + iterations):
        file_name = f'{input_path}{file_idx}.pkl'
        
        if not os.path.exists(file_name):
            print(f"File {file_name} does not exist. Skip...")
            continue
        
        with open(file_name, 'rb') as file:
            results_gt_pred = pickle.load(file)
        
        class_giou_sums = {i: 0.0 for i in range(len(CLASSES_DETR)) if CLASSES_DETR[i] != 'N/A'}
        class_counts = {i: 0 for i in range(len(CLASSES_DETR)) if CLASSES_DETR[i] != 'N/A'}

        for result in results_gt_pred:
            gt_bboxes = result.gt_instances.bboxes
            pred_bboxes = result.pred_instances.bboxes
            gt_labels = result.gt_instances.labels
            pred_labels = result.pred_instances.labels

            if gt_bboxes.numel() == 0 or pred_bboxes.numel() == 0:
                continue

            giou_matrix = generalized_box_iou(gt_bboxes, pred_bboxes)

            cost_matrix = 1 - giou_matrix.cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            for gt_idx, pred_idx in zip(row_ind, col_ind):
                gt_bbox = gt_bboxes[gt_idx]
                pred_bbox = pred_bboxes[pred_idx]
                gt_class = gt_labels[gt_idx].item()
                pred_class = pred_labels[pred_idx].item()

                if pred_class in class_mapping:
                    mapped_gt_class = class_mapping[pred_class]
                    if mapped_gt_class == gt_class:
                        giou_value = giou_matrix[gt_idx, pred_idx].item()
                        if mapped_gt_class in class_giou_sums:
                            class_giou_sums[mapped_gt_class] += giou_value
                            class_counts[mapped_gt_class] += 1

        average_giou_per_class = {}
        for cls, total_giou in class_giou_sums.items():
            count = class_counts[cls]
            if count > 0:
                average_giou_per_class[cls] = total_giou / count
            else:
                average_giou_per_class[cls] = None

        results = {CLASSES_DETR[cls]: avg_giou for cls, avg_giou in average_giou_per_class.items() if avg_giou is not None}

        output_file_name = f'bbox_regression_gt_pred_average_giou_per_class_result_{file_idx}.json'
        with open(output_path + output_file_name, 'w') as json_file:
            json.dump(results, json_file)

        print(f"Processing iteration: {file_idx+1}/{iterations}")

def calculate_average_f1_score_overall_and_save_results(input_path, output_path, iterations, start_idx=0):
    """
    This function processes a series of pickle files, extracts ground-truth and predicted classes, and calculates the F1 score for classification.

    Parameters:
    - input_path (str): The path where the pickle files are stored. The filename should include the index.
    Example: "ablation_studies/encoder_self_attention_5/integrated_ground_truth_encoder_self_attention_5_"
    - output_path (str): The path where the result files (JSON) should be stored.
    Example: "ablation_studies/encoder_self_attention_5/"
    - iterations (int): The number of files to process, starting from the start index.
    - start_idx (int): The starting index for the files. Default is 0.

    Returns:
    - The function saves the F1 score results as JSON files in the output path and prints the calculated values to the console.
    """
    
    for file_idx in range(start_idx, start_idx + iterations):
        file_path = f'{input_path}{file_idx}.pkl'
        
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist. Skip...")
            continue
        
        with open(file_path, 'rb') as file:
            data = pickle.load(file)

        mmdet_class_to_idx = {cls: idx for idx, cls in enumerate(CLASSES_MMDET)}

        all_gt_classes = []
        all_pred_classes = []

        for i in range(len(data)):
            gt_classes = data[i].gt_instances.labels.cpu()
            gt_bboxes = data[i].gt_instances.bboxes.cpu()
            pred_classes = data[i].pred_instances.labels.cpu()
            pred_scores = data[i].pred_instances.scores.cpu()
            pred_bboxes = data[i].pred_instances.bboxes.cpu()

            if len(gt_bboxes) == 0 or len(pred_bboxes) == 0:
                continue

            iou_matrix = generalized_box_iou(gt_bboxes, pred_bboxes).cpu().numpy()
            cost_matrix = 1 - iou_matrix
            row_indices, col_indices = linear_sum_assignment(cost_matrix)

            for row, col in zip(row_indices, col_indices):
                gt_class = gt_classes[row].item()
                pred_class = pred_classes[col].item()
                gt_class_name = CLASSES_DETR[gt_class] if gt_class < len(CLASSES_DETR) else 'N/A'
                pred_class_name = CLASSES_MMDET[pred_class] if pred_class < len(CLASSES_MMDET) else 'N/A'

                if gt_class_name in mmdet_class_to_idx and pred_class_name in mmdet_class_to_idx:
                    all_gt_classes.append(mmdet_class_to_idx[gt_class_name])
                    all_pred_classes.append(mmdet_class_to_idx[pred_class_name])
                else:
                    all_gt_classes.append(-1)
                    all_pred_classes.append(-1)

        valid_indices = [i for i, (gt, pred) in enumerate(zip(all_gt_classes, all_pred_classes)) if gt != -1 and pred != -1]
        valid_gt_classes = [all_gt_classes[i] for i in valid_indices]
        valid_pred_classes = [all_pred_classes[i] for i in valid_indices]

        f1 = f1_score(valid_gt_classes, valid_pred_classes, average='weighted')

        result = {
            "Overall_F1_Score": f1
        }

        output_file_path = f'f1_score_classification_overall_result_{file_idx}.json'
        with open(output_path + output_file_path, 'w') as json_file:
            json.dump(result, json_file)

        print(f"Processing iteration: {file_idx+1}/{iterations}")

def calculate_average_f1_score_per_class_and_save_results(input_path, output_path, iterations, start_idx=0):
    """
    This function processes a series of pickle files, extracts ground-truth and predicted classes, and calculates the F1 scores for each class.

    Parameters:
    - input_path (str): The path where the pickle files are stored. The filename should include the index.
    Example: "ablation_studies/encoder_self_attention_5/integrated_ground_truth_encoder_self_attention_5_"
    - output_path (str): The path where the result files (JSON) should be stored.
    Example: "ablation_studies/encoder_self_attention_5/"
    - iterations (int): The number of files to process, starting from the start index.
    - start_idx (int): The starting index for the files. Default is 0.

    Returns:
    - The function saves the F1 score results for each class as JSON files in the output path and prints the calculated values to the console.
    """
    
    for file_idx in range(start_idx, start_idx + iterations):
        file_path = f'{input_path}{file_idx}.pkl'
        
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist. Skip...")
            continue
        
        with open(file_path, 'rb') as file:
            data = pickle.load(file)

        mmdet_class_to_idx = {cls: idx for idx, cls in enumerate(CLASSES_MMDET)}

        all_gt_classes = []
        all_pred_classes = []

        for i in range(len(data)):
            gt_classes = data[i].gt_instances.labels.cpu()
            gt_bboxes = data[i].gt_instances.bboxes.cpu()
            pred_classes = data[i].pred_instances.labels.cpu()
            pred_scores = data[i].pred_instances.scores.cpu()
            pred_bboxes = data[i].pred_instances.bboxes.cpu()

            if len(gt_bboxes) == 0 or len(pred_bboxes) == 0:
                continue

            iou_matrix = generalized_box_iou(gt_bboxes, pred_bboxes).cpu().numpy()
            cost_matrix = 1 - iou_matrix
            row_indices, col_indices = linear_sum_assignment(cost_matrix)

            for row, col in zip(row_indices, col_indices):
                gt_class = gt_classes[row].item()
                pred_class = pred_classes[col].item()
                gt_class_name = CLASSES_DETR[gt_class] if gt_class < len(CLASSES_DETR) else 'N/A'
                pred_class_name = CLASSES_MMDET[pred_class] if pred_class < len(CLASSES_MMDET) else 'N/A'

                if gt_class_name in mmdet_class_to_idx and pred_class_name in mmdet_class_to_idx:
                    all_gt_classes.append(mmdet_class_to_idx[gt_class_name])
                    all_pred_classes.append(mmdet_class_to_idx[pred_class_name])
                else:
                    all_gt_classes.append(-1)
                    all_pred_classes.append(-1)

        valid_indices = [k for k, (gt, pred) in enumerate(zip(all_gt_classes, all_pred_classes)) if gt != -1 and pred != -1]
        valid_gt_classes = [all_gt_classes[k] for k in valid_indices]
        valid_pred_classes = [all_pred_classes[k] for k in valid_indices]

        class_f1_scores = {}
        for cls in CLASSES_MMDET:
            if cls in mmdet_class_to_idx:
                cls_idx = mmdet_class_to_idx[cls]
                gt_for_class = [1 if c == cls_idx else 0 for c in valid_gt_classes]
                pred_for_class = [1 if c == cls_idx else 0 for c in valid_pred_classes]
                f1 = f1_score(gt_for_class, pred_for_class, average='binary', zero_division=0)
                class_f1_scores[cls] = f1

        output_file_path = f'f1_score_classification_per_class_result_{file_idx}.json'
        with open(output_path + output_file_path, 'w') as json_file:
            json.dump(class_f1_scores, json_file)

        print(f"Processing iteration: {file_idx+1}/{iterations}")