import torch

def pretty_print_euclidean_distances_bbox_areas(euclidean_distances, bbox_areas):
    """
    Pretty prints the results of euclidean distances and bounding box areas.

    Parameters:
    euclidean_distances (torch.Tensor): Euclidean distances between bounding boxes.
    bbox_areas (torch.Tensor): Areas of bounding boxes.

    Returns:
    None
    """
    print("\nEUCLIDEAN DISTANCES")
    print("__________________________________________")

    half_len = len(euclidean_distances) // 2
    for i in range(half_len):
        left_euclidean = euclidean_distances[i].item()
        right_euclidean = euclidean_distances[i + half_len].item()
        print(f"Query {i}: {left_euclidean:.4f} \t Query {i + half_len}: {right_euclidean:.4f}")

    print("\nBOUNDING BOX AREAS")
    print("__________________________________________")

    for i in range(half_len):
        left_area = bbox_areas[i].item()
        right_area = bbox_areas[i + half_len].item()
        print(f"Query {i}: {left_area:.4f} \t Query {i + half_len}: {right_area:.4f}")


def pretty_print_zero_rows(zero_rows):
    """
    Pretty prints the zero rows (ablated rows).

    Parameters:
    zero_rows (list): List of indices representing the ablated rows.

    Returns:
    None
    """
    print("\nABLATED ROWS")
    print("______________________________________")
    if zero_rows:
        half_len = (len(zero_rows) + 1) // 2
        first_column = zero_rows[:half_len]
        second_column = zero_rows[half_len:]
        for i in range(half_len):
            left_row_str = f"Ablation Row: {first_column[i]}"
            right_row_str = f"Ablation Row: {second_column[i]}" if i < len(second_column) else ""
            print(f"{left_row_str:<30} {right_row_str}")
    else:
        print("No Ablations found.")

def pretty_print_topk_50_class_changes(changes, CLASSES):
    """
    Pretty prints the top 50 class changes.

    Parameters:
    changes (torch.Tensor): Matrix representing the changes between classes predicted by two models.
    CLASSES (list): List of class labels.

    Returns:
    None
    """
    top_changes = torch.flatten(changes).topk(50)
    top_change_indices = [(idx // 80, idx % 80) for idx in top_changes.indices]

    row_format = "{:<3} | {:<20} | {:<20} | {:<10}"

    print(row_format.format("No.", "From Class", "To Class", "Count"))
    print("-" * 60)

    for idx, (i, j) in enumerate(top_change_indices):
        print(row_format.format(
            idx+1, 
            CLASSES[i], 
            CLASSES[j], 
            f"{top_changes.values[idx].item():.0f}"
        ))