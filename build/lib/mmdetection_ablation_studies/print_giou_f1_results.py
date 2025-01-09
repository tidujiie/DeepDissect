import json
import pandas as pd
import numpy as np
import os

def print_and_save_aggregated_giou_overall_results(baseline_file_path, ablations_file_path_template, output_path, iterations, start_idx=0):
    """
    This function calculates and aggregates the average GIoU values from a baseline JSON file and multiple ablation study JSON files.
    
    Parameters:
    - baseline_path (str): Path to the baseline GIoU JSON file.
    - ablations_path (str): Template path for the ablation study JSON files with a placeholder for indices.
    - output_path (str): Directory path to save the results CSV file.
    - iterations (int): Number of ablation study files to process.
    - start_idx (int): Starting index for the ablation files (default is 0).
    
    Returns:
    - Saves the aggregated results as a CSV file in the specified output directory and prints the results to the console.
    """
    giou_values = []

    with open(baseline_file_path, 'r') as baseline_file:
        baseline_data = json.load(baseline_file)
        baseline_giou = baseline_data["average_giou"]

    for i in range(start_idx, iterations):
        file_path = ablations_file_path_template.format(i)
        with open(file_path, 'r') as file:
            data = json.load(file)
            giou_values.append(data["average_giou"])

    average_giou = sum(giou_values) / len(giou_values)
    std_dev = np.std(giou_values)
    delta = average_giou - baseline_giou

    results_df = pd.DataFrame({
        'Baseline GIoU': [baseline_giou],
        'Ablations AVG GIoU': [average_giou],
        'Delta': [delta],
        'Ablations Stdev': [std_dev]
    })

    print(results_df.to_string(index=False))
    results_df.to_csv(output_path + "bbox_giou_overall_results.csv", index=False)


def print_and_save_giou_per_class_results(baseline_file_path, ablations_file_path_template, output_path, iterations, start_idx):
    """
    This function calculates the average and standard deviation of GIoU (Generalized Intersection over Union) for 
    each class from multiple ablation study result files. It compares these values to baseline GIoU values and 
    computes the delta (difference). The results, including the baseline GIoU, average GIoU, delta GIoU, and 
    standard deviation, are then saved as a CSV file.

    Parameters:
    - baseline_input_path (str): Path to the JSON file containing the baseline GIoU per class.
    - ablations_path_template (str): Template string for ablation study JSON file paths, where '{}' will be replaced with the file index.
    - output_path (str): Path where the resulting CSV file will be saved.
    - iterations (int): Number of files to iterate over in the ablation study directory.
    - start_idx (int): The starting index for the ablation study files.
    """

    with open(baseline_file_path, 'r') as file:
        baseline_data = json.load(file)

    baseline_df = pd.DataFrame(list(baseline_data.items()), columns=['Class', 'Baseline GIoU'])

    giou_dict = {cls: [] for cls in baseline_data.keys()}

    for i in range(start_idx, iterations):
        file_path = ablations_file_path_template.format(i)
        with open(file_path, 'r') as file:
            data = json.load(file)
            for cls, giou in data.items():
                giou_dict[cls].append(giou)

    average_giou = {cls: np.mean(giou_list) for cls, giou_list in giou_dict.items()}
    std_giou = {cls: np.std(giou_list) for cls, giou_list in giou_dict.items()}
    delta_giou = {cls: average_giou[cls] - baseline_data[cls] for cls in average_giou.keys()}

    average_giou_df = pd.DataFrame(list(average_giou.items()), columns=['Class', 'Ablations AVG GIoU'])
    std_giou_df = pd.DataFrame(list(std_giou.items()), columns=['Class', 'Ablations Stdev'])
    delta_giou_df = pd.DataFrame(list(delta_giou.items()), columns=['Class', 'Delta'])

    result_df = pd.merge(baseline_df, average_giou_df, on='Class')
    result_df = pd.merge(result_df, delta_giou_df, on='Class')
    result_df = pd.merge(result_df, std_giou_df, on='Class')

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print(result_df.to_string(index=False))

    result_df.to_csv(output_path + 'giou_score_per_class_results.csv', index=False)


def print_and_save_aggregated_f1_overall_results(baseline_file_path, ablations_file_path_template, output_path, iterations, start_idx=0):
    """
    This function calculates and aggregates the average f1 values from a baseline JSON file and multiple ablation study JSON files.
    
    Parameters:
    - baseline_path (str): Path to the baseline f1 JSON file.
    - ablations_path (str): Template path for the ablation study JSON files with a placeholder for indices.
    - output_path (str): Directory path to save the results CSV file.
    - iterations (int): Number of ablation study files to process.
    - start_idx (int): Starting index for the ablation files (default is 0).
    
    Returns:
    - Saves the aggregated results as a CSV file in the specified output directory and prints the results to the console.
    """
    f1_values = []

    with open(baseline_file_path, 'r') as baseline_file:
        baseline_data = json.load(baseline_file)
        baseline_f1 = baseline_data["Overall_F1_Score"]

    for i in range(start_idx, iterations):
        file_path = ablations_file_path_template.format(i)
        with open(file_path, 'r') as file:
            data = json.load(file)
            f1_values.append(data["Overall_F1_Score"])

    average_f1 = sum(f1_values) / len(f1_values)
    std_dev = np.std(f1_values)
    delta = average_f1 - baseline_f1

    results_df = pd.DataFrame({
        'Baseline F1': [baseline_f1],
        'Ablations AVG F1': [average_f1],
        'Delta': [delta],
        'Ablations Stdev': [std_dev]
    })

    print(results_df.to_string(index=False))
    results_df.to_csv(output_path + "bbox_f1_overall_results.csv", index=False)


def print_and_save_f1_per_class_results(baseline_file_path, ablations_file_path_template, output_path, iterations, start_idx):
    """
    This function calculates the average and standard deviation of F1 Scores for 
    each class from multiple ablation study result files. It compares these values to baseline F1 values and 
    computes the delta (difference). The results, including the baseline F1 Score, average F1 Score, delta F1 Score, and 
    standard deviation, are then saved as a CSV file.

    Parameters:
    - baseline_input_path (str): Path to the JSON file containing the baseline F1 Score per class.
    - ablations_path_template (str): Template string for ablation study JSON file paths, where '{}' will be replaced with the file index.
    - output_path (str): Path where the resulting CSV file will be saved.
    - iterations (int): Number of files to iterate over in the ablation study directory.
    - start_idx (int): The starting index for the ablation study files.
    """

    with open(baseline_file_path, 'r') as file:
        baseline_data = json.load(file)

    baseline_df = pd.DataFrame(list(baseline_data.items()), columns=['Class', 'Baseline F1'])

    f1_dict = {cls: [] for cls in baseline_data.keys()}

    for i in range(start_idx, iterations):
        file_path = ablations_file_path_template.format(i)
        with open(file_path, 'r') as file:
            data = json.load(file)
            for cls, f1 in data.items():
                f1_dict[cls].append(f1)

    average_f1 = {cls: np.mean(f1_list) for cls, f1_list in f1_dict.items()}
    std_f1 = {cls: np.std(f1_list) for cls, f1_list in f1_dict.items()}
    delta_f1 = {cls: average_f1[cls] - baseline_data[cls] for cls in average_f1.keys()}

    average_f1_df = pd.DataFrame(list(average_f1.items()), columns=['Class', 'Ablations AVG F1 Score'])
    std_f1_df = pd.DataFrame(list(std_f1.items()), columns=['Class', 'Ablations Stdev'])
    delta_f1_df = pd.DataFrame(list(delta_f1.items()), columns=['Class', 'Delta'])

    result_df = pd.merge(baseline_df, average_f1_df, on='Class')
    result_df = pd.merge(result_df, delta_f1_df, on='Class')
    result_df = pd.merge(result_df, std_f1_df, on='Class')

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    print(result_df.to_string(index=False))

    result_df.to_csv(output_path + 'f1_score_per_class_results.csv', index=False)