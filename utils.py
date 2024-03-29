import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import exposure
from torch.utils.data import DataLoader

from ni_dataset import NaturalnessDataset
from pl_trainer import NaturalnessTrainer

import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable


WC_palette = {
    10: (0, 160, 0),  # "Tree cover" 00a000
    20: (150, 100, 0),  # "Shrubland" 966400
    30: (255, 180, 0),  # "Grassland" ffb400
    40: (255, 255, 100),  # "Cropland" ffff64
    50: (195, 20, 0),  # "Built-up" c31400
    60: (255, 245, 215),  # "Bare / sparse vegetation" fff5d7
    70: (255, 255, 255),  # "Snow and ice" ffffff
    80: (0, 70, 200),  # "Permanent water bodies" 0046c8
    90: (0, 220, 130),  # "Herbaceous wetland" 00dc82
    95: (0, 150, 120),  # "Mangroves" 009678
    100: (255, 235, 175),
}  # "Moss and lichen" ffebaf

cmappx = [
    "#00a000",
    "#966400",
    "#ffb400",
    "#ffff64",
    "c31400",
    "#fff5d7",
    "#ffffff",
    "#0046c8",
    "#00dc82",
    "#009678",
    "#ffebaf",
]

rgb = [
    (0, 160, 0),
    (150, 100, 0),
    (255, 180, 0),
    (255, 255, 100),
    (195, 20, 0),
    (255, 245, 215),
    (255, 255, 255),
    (0, 70, 200),
    (0, 220, 130),
    (0, 150, 120),
    (255, 235, 175),
]


modality_lookup = {
    "s2": 10,
    "s1": 2,
    "esa_wc": 1,
    "viirs": 1,
    "all": 14,
}


def load_json(json_path):
    # Load the arguments
    with open(json_path, "r") as f:
        config = json.load(f)
    return config


def prepare_test_dataset_loader_model(config):
    # Initialize the test dataset
    test_dataset = NaturalnessDataset(
        split_file=config["split_file"],
        crop_size=config["crop_size"],
        root=config["data_root"],
        split="test",
        transforms=None,
        return_modality=config["return_modality"],
    )

    # Initialize the test data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=False,
        drop_last=False,
    )

    # Assuming we want to load the model from a checkpoint for testing
    checkpoint_path = config["checkpoint_path"]
    if checkpoint_path:
        model = NaturalnessTrainer.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            input_size=(config["crop_size"], config["crop_size"]),
            n_input_ch=modality_lookup[config["return_modality"]],
            learning_rate=config["learning_rate"],
            batch_size=config["batch_size"],
            use_lr_scheduler=config["use_lr_scheduler"],
            log_distributions_flag=config["log_distributions_flag"],
            use_embedding_occlusion=config["use_embedding_occlusion"],
            exp_name=config["exp_name"],
            occlusion_type=config["occlusion_type"],
        )
    else:
        # Initialize the model normally if there is no checkpoint
        model = NaturalnessTrainer(
            input_size=(config["crop_size"], config["crop_size"]),
            n_input_ch=modality_lookup[config["return_modality"]],
            learning_rate=config["learning_rate"],
            batch_size=config["batch_size"],
            use_lr_scheduler=config["use_lr_scheduler"],
            log_distributions_flag=config["log_distributions_flag"],
            use_embedding_occlusion=config["use_embedding_occlusion"],
            exp_name=config["exp_name"],
            occlusion_type=config["occlusion_type"],
        )
    return test_dataset, test_loader, model


def transform_data(data):
    """
    Transforms the data from tensor format to a dictionary with wdpa_id as keys
    and their corresponding values as lists of floats.
    """
    transformed_data = {}
    for item in data:
        # Convert wdpa_id from tensor to int
        wdpa_id = item["wdpa_id"].item()
        diffs = item["diffs"]

        # Convert each tensor in diffs to a float and wrap in a list
        transformed_diffs = {key: [value.item()] for key, value in diffs.items()}
        transformed_data[wdpa_id] = transformed_diffs
    return transformed_data


def calculate_modality_means(transformed_data):
    """
    Calculates the mean across wdpa_ids for each modality from the transformed data.
    """
    modality_means = {"S2": [], "S1": [], "WC": [], "VIIRS": [], "ALL": []}

    # Accumulate values for each modality across all wdpa_ids
    for diffs in transformed_data.values():
        for modality, values in diffs.items():
            modality_means[modality].append(values[0])

    # Calculate mean for each modality
    modality_mean_values = {
        modality: np.mean(values) for modality, values in modality_means.items()
    }
    return modality_mean_values


def plot_means(values, test=False):

    if test:
        modality_indices = {
            "S2": list(range(0, 10)),
            "S1": [10, 11],
            "WC": [12],
            "VIIRS": [13],
            "ALL": list(range(0, 14)),
        }
    else:
        modality_indices = {
            "S2": list(range(0, 10)),
            "S1": [10, 11],
            "WC": [12],
            "VIIRS": [13],
        }

    plt.figure(figsize=(10, 6))
    plt.bar(modality_indices.keys(), values, color="skyblue", edgecolor="black")
    plt.xlabel("Bands")
    plt.ylabel("Values")

    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)

    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.show()
    print(values)


def stretch(im):
    p1, p99 = np.percentile(im, (1, 99))
    J = exposure.rescale_intensity(im, in_range=(p1, p99))
    J = J / J.max()
    return J


def get_channels(im, all_channels, select_channels):
    """
    Filters the channels for a given set of channels. Used to extract RGB channels from full-spectral S2 image.
    """

    all_ch = all_channels
    final_ch = select_channels

    chns = [
        all_ch.index(final_ch) if isinstance(final_ch, str) else final_ch
        for final_ch in final_ch
    ]

    return im[chns]

def convert_to_color_wc(arr_2d, palette=WC_palette):
    """Numeric labels to RGB-color encoding."""
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d


cmap_colors = ["#d7191c", "#fdae61", "#ffffc0", "#a6d96a", "#1a9641"]

ni_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    name="ni_map", colors=cmap_colors
)


def integer_decode(normalized_array):
    min_val = 10.0
    max_val = 100.0
    original_array = normalized_array * (max_val - min_val) + min_val
    return original_array


def get_data_viz(test_dataset, model, idx):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bands = (
        "B2",
        "B3",
        "B4",
        "B5",
        "B6",
        "B7",
        "B8",
        "B8A",
        "B11",
        "B12",
        "VV",
        "VH",
        "LC",
        "VIIRS",
    )
    image_mask = test_dataset[idx]
    # wdpa_id = image_mask[-1]
    input = torch.Tensor(np.expand_dims(image_mask[0]["image"], 0)).to(device)

    logits_reg = model(input)
    logits_reg = logits_reg["pred_act"]

    image = image_mask[0]["image"]
    gt_mask = image_mask[0]["mask"]
    rgb_s2 = get_channels(
        im=image, all_channels=bands, select_channels=["B4", "B3", "B2"]
    )
    s1 = get_channels(im=image, all_channels=bands, select_channels=["VH"])
    esa_wc = get_channels(im=image, all_channels=bands, select_channels=["LC"])
    viirs = get_channels(im=image, all_channels=bands, select_channels=["VIIRS"])

    s2_im = stretch(np.transpose(rgb_s2.numpy(), (1, 2, 0)))

    gt_mask = gt_mask.squeeze()
    gt_mask_numpy = np.array(gt_mask)

    pr_mask = logits_reg.detach().cpu().numpy().squeeze()
    pr_mask = pr_mask.astype(np.float32)

    nan_indices = np.isnan(gt_mask_numpy)
    pr_mask[nan_indices] = np.nan

    esa_wc_inteight = np.asarray(esa_wc, dtype="uint8")
    esa_wc_inteight = np.rint(integer_decode(esa_wc))
    esa_color = convert_to_color_wc(esa_wc_inteight.squeeze())

    data = {
        "s2": s2_im,
        "s1": s1,
        "esa": esa_color,
        "gt": gt_mask,
        "pred": pr_mask,
        "viirs": viirs,
    }
    return data


def viz_bar_images(idx, bar_vals, data, exp_id=None):

    if len(bar_vals) == 4:
        categories = ("Sentinel-2", "Sentinel-1", "Land Cover", "Night Lights")
    if len(bar_vals) == 5:
        categories = ("Sentinel-2", "Sentinel-1", "Land Cover", "Night Lights", "ALL")

    fig = plt.figure(figsize=(16, 8))

    # Create a GridSpec layout
    gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 1])

    # Create the bar plot
    ax0 = plt.subplot(gs[:, 0])
    ax0.bar(categories, bar_vals, color="orange")
    ax0.set_title("")
    # ax0.set_ylim([values.min(),values.max()])
    ax0.set_xlabel("Occluded Modalities")
    ax0.set_ylabel("Distance between the Occluded and Non-Occluded Embeddings")
    ax0.set_xticks(np.arange(len(categories)))
    ax0.set_xticklabels(categories, rotation=45)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.spines["bottom"].set_visible(False)
    ax0.spines["left"].set_visible(False)

    # Plot data
    ax1 = plt.subplot(gs[0, 1])
    ax1.imshow(data["s2"])
    ax1.axis("off")
    ax1.set_title("Sentinel-2")

    ax2 = plt.subplot(gs[0, 2])
    ax2.imshow(data["esa"])
    ax2.axis("off")
    ax2.set_title("Land Cover")

    ax3 = plt.subplot(gs[0, 3])
    im3 = ax3.imshow(data["gt"], vmin=0, vmax=1, cmap=ni_cmap, interpolation="none")
    ax3.axis("off")
    ax3.set_title("Land Naturalness Annotation")

    ax4 = plt.subplot(gs[1, 1])
    ax4.imshow(data["s1"].squeeze())
    ax4.axis("off")
    ax4.set_title("Sentinel-1")

    ax5 = plt.subplot(gs[1, 2])
    im5 = ax5.imshow(data["viirs"].squeeze(), vmin=0, vmax=1, cmap="BuPu")
    ax5.axis("off")
    ax5.set_title("Night Lights")

    ax6 = plt.subplot(gs[1, 3])
    im6 = ax6.imshow(data["pred"], vmin=0, vmax=1, cmap=ni_cmap, interpolation="none")
    ax6.axis("off")
    ax6.set_title("Land Naturalness Prediction")

    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im3, cax=cax3)

    divider5 = make_axes_locatable(ax5)
    cax5 = divider5.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im5, cax=cax5)

    divider6 = make_axes_locatable(ax6)
    cax6 = divider6.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im6, cax=cax6)

    plt.tight_layout()

    if exp_id is not None:
        save_path = f"./logs/{exp_id}__regularizer/viz_preds_bars"
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"{idx}.png"))

    plt.show()


def viz_images(idx, data, exp_id=None):

    fig = plt.figure(figsize=(16, 8))

    # Create a GridSpec layout
    gs = gridspec.GridSpec(2, 4, width_ratios=[1, 1, 1, 1])

    # Plot data
    ax1 = plt.subplot(gs[0, 1])
    ax1.imshow(data["s2"])
    ax1.axis("off")
    ax1.set_title("Sentinel-2")

    ax2 = plt.subplot(gs[0, 2])
    ax2.imshow(data["esa"])
    ax2.axis("off")
    ax2.set_title("Land Cover")

    ax3 = plt.subplot(gs[0, 3])
    im3 = ax3.imshow(data["gt"], cmap=ni_cmap, vmin=0, vmax=1, interpolation="none")
    ax3.axis("off")
    ax3.set_title("Land Naturalness Annotation")

    ax4 = plt.subplot(gs[1, 1])
    ax4.imshow(data["s1"].squeeze())
    ax4.axis("off")
    ax4.set_title("Sentinel-1")

    ax5 = plt.subplot(gs[1, 2])
    im5 = ax5.imshow(data["viirs"].squeeze(), vmin=0, vmax=1, cmap="BuPu")
    ax5.axis("off")
    ax5.set_title("Night Lights")

    ax6 = plt.subplot(gs[1, 3])
    im6 = ax6.imshow(data["pred"], cmap=ni_cmap, vmin=0, vmax=1, interpolation="none")
    ax6.axis("off")
    ax6.set_title("Land Naturalness Prediction")

    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im3, cax=cax3)

    divider5 = make_axes_locatable(ax5)
    cax5 = divider5.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im5, cax=cax5)

    divider6 = make_axes_locatable(ax6)
    cax6 = divider6.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im6, cax=cax6)

    plt.tight_layout()

    if exp_id is not None:
        save_path = f"./logs/{exp_id}__regularizer/viz_preds_bars"
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"{idx}.png"))

    plt.show()
