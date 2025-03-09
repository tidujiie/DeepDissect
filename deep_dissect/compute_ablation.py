import torch
import numpy as np
import os
import random
from copy import deepcopy
from deep_dissect.utility import get_model_state_dict_and_keys

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


def compute_single_wise_ablations_percentage(
        PATH_CHECKPOINT_FILE,
        ablation_percentage,
        ablation_component,
        number_of_ablations=1,
        PATH_SAVE_DIR=''):
    """
    This function performs ablations on a specific component of a neural network model stored in a checkpoint file.
    Ablation is done by setting a randomly sampled percentage of the component's values to zero, simulating the effect
    of removing them from the model. This process is repeated for a specified number of iterations, each time with a
    different random set of values ablated. Each ablated version of the model is then saved to the specified directory.

    Parameters:
    - PATH_CHECKPOINT_FILE (str): Path to the model's checkpoint file.
    - ablation_percentage (int): The percentage of the component's values to be ablated.
    - ablation_component (str): The name of the component to be ablated (e.g., a specific layer or weight matrix).
    - number_of_ablations (int): The number of ablated versions of the model to generate.
    - PATH_SAVE_DIR (str): The directory where the ablated models will be saved.
    """
    state_dict, weight_keys = get_model_state_dict_and_keys(PATH_CHECKPOINT_FILE)

    if ablation_component not in weight_keys:
        raise ValueError(
            f"Component '{ablation_component}' incorrect or not specified. Available options are: {', '.join(weight_keys)}.")

    for i in range(number_of_ablations):
        model = torch.load(PATH_CHECKPOINT_FILE)
        component = model['state_dict'][ablation_component]

        total_elements = component.numel()
        elements_to_ablate = int(total_elements * ablation_percentage / 100)

        indices = torch.randperm(total_elements)[:elements_to_ablate]

        flat_component = component.flatten()
        flat_component[indices] = 0
        component_ablated = flat_component.view(component.shape)

        model['state_dict'][ablation_component] = component_ablated
        save_path = f"{PATH_SAVE_DIR}model_{ablation_component}_single_wise_{ablation_percentage}_{i}.pth"
        torch.save(model, save_path)


def compute_attention_head_detr_ablations_percentage(
        PATH_CHECKPOINT_FILE,
        ablation_percentage=5,
        ablation_component_template="encoder.layers.{layer}.self_attn.attn.in_proj_weight",
        layers_to_ablate=range(6),
        number_of_ablations=1,
        PATH_SAVE_DIR=''):
    """
    This function performs targeted ablations on attention heads within a specified model by setting a randomly sampled
    percentage of  their parameters to zero. The ablation is applied to the specified layers of the model's attention
    mechanism, specifically to the input projection weights.

    Parameters:
    - PATH_CHECKPOINT_FILE (str): Path to the model's checkpoint file.
    - ablation_percentage (int): The percentage of the component's values to be ablated.
    - ablation_component_template (str): The name of the component to be ablated with placeholder for iteration.
    - layers_to_ablate (list/range): The layers to ablate (e.g., range(6) for layers 0 to 5).
    - number_of_ablations (int): The number of ablated versions of the model to generate.
    - PATH_SAVE_DIR (str): The directory where the ablated models will be saved.
    """
    model = torch.load(PATH_CHECKPOINT_FILE)

    sample_component = ablation_component_template.format(layer=layers_to_ablate[0])
    first_dim = model['state_dict'][sample_component].size(0)
    total_heads = 8 * first_dim
    heads_per_row = 8
    total_heads_to_ablate = int(total_heads * (ablation_percentage / 100))

    if PATH_SAVE_DIR and not os.path.exists(PATH_SAVE_DIR):
        os.makedirs(PATH_SAVE_DIR)

    for i in range(number_of_ablations):
        ablated_checkpoint = deepcopy(model)

        for layer in layers_to_ablate:
            ablation_component = ablation_component_template.format(layer=layer)
            component = ablated_checkpoint['state_dict'][ablation_component]

            weights = component
            modified_weights = weights.clone()

            head_indices_to_ablate = random.sample(range(total_heads), total_heads_to_ablate)

            for head_index in head_indices_to_ablate:
                row = head_index // heads_per_row
                head_in_row = head_index % heads_per_row
                start_col = head_in_row * 32
                end_col = start_col + 32
                modified_weights[row, start_col:end_col] = 0

            ablated_checkpoint['state_dict'][ablation_component] = modified_weights

        save_path = os.path.join(PATH_SAVE_DIR, f"model_" + ablation_component_template.replace(".{layer}",
                                                                                                "") + f"_attention_head_{ablation_percentage}_{i}.pth")
        torch.save(ablated_checkpoint, save_path)
        print(f"Iteration {i + 1} / {number_of_ablations}")


def compute_component_wise_complete_ablations(
        PATH_CHECKPOINT_FILE,
        ablation_component,
        PATH_SAVE_DIR=''):
    """
    This function performs a complete ablation of a specified component within a mmdetection model. The complete
    component is ablated.

    Parameters:
    - PATH_CHECKPOINT_FILE (str): Path to the model's checkpoint file.
    - ablation_component (str): The name of the component to be ablated (e.g., a specific layer or weight matrix).
    - PATH_SAVE_DIR (str): The directory where the ablated models will be saved.
    """
    model = torch.load(PATH_CHECKPOINT_FILE)

    if PATH_SAVE_DIR and not os.path.exists(PATH_SAVE_DIR):
        os.makedirs(PATH_SAVE_DIR)

    ablated_checkpoint = deepcopy(model)

    ablated_checkpoint['state_dict'][ablation_component] = torch.zeros_like(
        ablated_checkpoint['state_dict'][ablation_component])

    save_path = f"{PATH_SAVE_DIR}model_{ablation_component}_component_wise.pth"
    torch.save(ablated_checkpoint, save_path)


def compute_progressive_component_ablations(
        PATH_CHECKPOINT_FILE,
        base_component_path="encoder.layers",
        component_to_ablate="self_attn.attn.in_proj_weight",
        PATH_SAVE_DIR=''):
    """
    This function performs a cumulative ablation of a specified component progressively on all layers of a mmdetection model.
    Ablation is carried out incrementally, starting from the first layer up to the final layer, with each resulting 
    model saved separately.

    Parameters:
    - PATH_CHECKPOINT_FILE (str): Path to the model's checkpoint file.
    - base_component_path (str): Base path in the state_dict before the layer index and component.
    - component_to_ablate (str): The specific component after the layer index to be ablated.
    - PATH_SAVE_DIR (str): Directory to save the ablated model checkpoints.
    """
    model = torch.load(PATH_CHECKPOINT_FILE)

    if PATH_SAVE_DIR and not os.path.exists(PATH_SAVE_DIR):
        os.makedirs(PATH_SAVE_DIR)

    total_layers = 0
    while True:
        layer_key = f"{base_component_path}.{total_layers}.{component_to_ablate}"
        if layer_key in model['state_dict']:
            total_layers += 1
        else:
            break

    for layer in range(total_layers):
        ablated_checkpoint = deepcopy(model)
        for l in range(layer + 1):
            layer_key = f"{base_component_path}.{l}.{component_to_ablate}"
            ablated_checkpoint['state_dict'][layer_key] = torch.zeros_like(
                ablated_checkpoint['state_dict'][layer_key])

        save_path = f"{PATH_SAVE_DIR}model_{base_component_path}_{layer}_{component_to_ablate}_component_layers.pth"
        torch.save(ablated_checkpoint, save_path)
