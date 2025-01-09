import matplotlib.pyplot as plt
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
from IPython.display import display
from ipywidgets import widgets
import numpy as np

def visualize_query_distribution(preds, zero_rows=None):
    """
    Visualizes the distribution of queries by plotting bounding boxes on a grid of subplots.

    Parameters:
    preds (torch.Tensor): A tensor of predicted bounding boxes with shape [M, N, 4], where M is the number of instances,
                          N is the number of queries (fixed to 2*n in this function), and 4 represents the bounding box
                          coordinates (cx, cy, width, height).
    zero_rows (list or None): A list of query indices to highlight in red. If None, no queries are highlighted.

    Returns:
    list: A list of color tensors representing the colors used for plotting the bounding boxes.
    """
    color_list = []
    s = (25, 10)
    fig = plt.figure(figsize=s)
    n = 10

    for idx, query in enumerate(range(n * 2), 1):
        ax = fig.add_subplot(4, n, idx)
        p = preds[:, query]
        assert p.min() >= 0
        assert p.max() <= 1
        cx, cy, w, h = p.unbind(-1)
        area = (w * h) ** 0.5 * 10
        color = (w * h) ** 0.5
        color = torch.stack((w, 1 - color, h), 1)
        color_list.append(color)
        plt.scatter(cx, cy, c=color, s=area, alpha=0.75)
        plt.xticks([])
        plt.yticks([])
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        if zero_rows is not None and query in zero_rows:
            title_color = 'red'
        elif zero_rows is not None and query not in zero_rows:
            title_color = 'green'
        else:
            title_color = 'black'

        ax.set_title(f"Query {query}", color=title_color)
        
    fig.tight_layout()
    plt.show()
    return color_list

def visualize_vector_field_distributions_query_ids(preds_baseline, preds_ablations, color_list):
    """
    Visualizes the distributions of vector fields for a given set of query IDs using quiver plots.

    Parameters:
    preds_baseline (torch.Tensor): Predictions for the baseline model with shape [M, N, 4], where M is the number of instances,
                                   N is the number of queries, and 4 represents the bounding box coordinates (cx, cy, width, height).
    preds (torch.Tensor): Predictions for the model to compare with the baseline, with the same shape as preds_baseline.
    color_list (list): A list of color tensors representing the colors used for plotting the vector fields for the baseline model, comes from query_distribution.

    Returns:
    None
    """
    num_queries = 20

    fig, axs = plt.subplots(4, 5, figsize=(40, 32))
    axs = axs.flatten()

    for query_index in range(num_queries):
        p = preds_baseline[:, query_index].sigmoid().cpu()
        q = preds_ablations[:, query_index].sigmoid().cpu()
        co = color_list[query_index]

        cx, cy, w, h = p.unbind(-1)
        cu, cv, w2, h2 = q.unbind(-1)

        axs[query_index].quiver(cx, cy, cu-cx, cv-cy, color=co.numpy(), scale=None, scale_units='xy')
        axs[query_index].set_title(f'Query {query_index}')

    plt.tight_layout()
    plt.show()
