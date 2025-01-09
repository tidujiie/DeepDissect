import torch
import numpy as np
import os

PATH_CHECKPOINT_FILE = 'C:/Users/Florian Hölken/source/repos/mmdetection/checkpoints/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth'

def ablate_query_embedding_percentage(query_embedding, ablation_percentage):
    """
    Ablate random single units in the query embedding based on a percentage.

    Parameters:
    - query_embedding (torch.Tensor): The query embedding tensor of shape (100, 256).
    - ablation_percentage (float): Percentage of units to ablate.

    Returns:
    - torch.Tensor: Ablated query embedding.
    """
    ablation_percentage = max(0, min(ablation_percentage, 100))

    total_units = query_embedding.numel()
    num_ablations = int((ablation_percentage / 100) * total_units)

    flattened = query_embedding.view(-1)

    indices_to_ablate = torch.randperm(flattened.size(0))[:num_ablations]

    flattened[indices_to_ablate] = 0

    return flattened.view_as(query_embedding)

def compute_single_wise_ablations_percentage(PATH_CHECKPOINT_FILE, ablation_percentage, ablation_component, number_of_ablations, PATH_SAVE_DIR):
    """
    Perform ablation on a specified component of a model's weights by a certain percentage,
    repeated for a specified number of iterations, and save each ablated model.

    Parameters:
    - PATH_CHECKPOINT_FILE (str): The file path to the model's checkpoint.
    - ablation_percentage (float): The percentage of the component to be ablated.
    - ablation_component (str): The specific component of the model to ablate.
    - number_of_ablations (int): The number of times to repeat the ablation process.
    - PATH_SAVE_DIR (str): The directory path where the ablated models will be saved.

    Save:
    - Saves the ablated states of the tensors and the ablation maps
    """
    weight_keys = get_model_state_dict_weight(PATH_CHECKPOINT_FILE)
    
    if ablation_component not in weight_keys:
        raise ValueError(f"Component '{ablation_component}' incorrect or not specified. Available options are: {', '.join(weight_keys)}.")

    for i in range(number_of_ablations):
        model = torch.load(PATH_CHECKPOINT_FILE)

        model_ablation_component = model['state_dict'][ablation_component]
        ablated_query_embedding = ablate_query_embedding_percentage(model_ablation_component, ablation_percentage)
        
        model['state_dict'][ablation_component] = ablated_query_embedding
        
        save_path = f"{PATH_SAVE_DIR}detr_{ablation_component}_single_wise_{ablation_percentage}_{i}.pth"
        torch.save(model, save_path)
        
def get_model_state_dict_weight(PATH_CHECKPOINT_FILE):
    """
    Retrieve the keys of weight tensors from a model's state dictionary that end with 'weight'.
    This is useful for identifying which components of the model contain trainable weights,
    particularly when planning for operations such as ablation or pruning.

    Parameters:
    - PATH_CHECKPOINT_FILE (str): The path to the checkpoint file of the model. This file
      contains the model's state dictionary among other information, and is typically saved
      using `torch.save()`.

    Returns:
    - weight_keys (list of str): A list of keys from the model's state dictionary that
      correspond to weight tensors. These keys are used to identify weight parameters
      within the model, enabling targeted manipulations like weight ablation.

    """
    model = torch.load(PATH_CHECKPOINT_FILE)
    state_dict = model['state_dict']
    
    weight_keys = []
    
    for key in state_dict.keys():
        if key.endswith('weight'):
            weight_keys.append(key)
    
    return weight_keys

def save_model_ablation_map(PATH_CHECKPOINT_FILE, ablation_component):
    """
    Save an ablation map for a specific component of a model's state dictionary to a new file.
    The ablation map is a binary tensor where each value is set to 1 if the original value in the
    state dictionary's component was 0, otherwise, it's set to 0. This map can be used for further
    analysis or modification of the model.

    Parameters:
    - PATH_CHECKPOINT_FILE (str): The path to the model's checkpoint file. This file should
      contain the model's state dictionary, saved with `torch.save()`.
    - ablation_component (str): The key within the model's state dictionary for which the
      ablation map will be generated. This key should correspond to a tensor within the state
      dictionary.

    Returns:
    None. The function saves the generated ablation map to a file with a modified name of the
    original checkpoint file, inserting '_ablationmap' appropriately to match the desired naming convention.
    """
    model = torch.load(PATH_CHECKPOINT_FILE)
    state_dict_ablation_component = model['state_dict'][ablation_component]

    ablation_map = (state_dict_ablation_component == 0).int()

    base_path, extension = os.path.splitext(PATH_CHECKPOINT_FILE)
    parts = base_path.rsplit('_', 2) 
    new_base_path = '_'.join(parts[:-2]) + '_ablationmap_' + '_'.join(parts[-2:])

    save_path = new_base_path + extension

    torch.save(ablation_map, save_path)

def get_ablationmap(PATH_ABLATIONMAP_FILE):
    model = torch.load(PATH_ABLATIONMAP_FILE)
    print(model)