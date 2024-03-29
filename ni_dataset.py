# +
import os

import numpy as np
import pandas as pd
import rasterio
import torch
from torch import Tensor

VV_STATS = {"mean": -11.057752150501129, "std": 3.739166317370456}
VH_STATS = {"mean": -18.362661381113668, "std": 4.6393440257502085}
AVG_RAD_STATS = {"mean": 2.0381726408726353, "std": 12.640913237652645}


class NaturalnessDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split_file,
        crop_size,
        root: str = "data",
        split: str = "train",
        transforms=None,
        seed=31,
        return_modality="all",
    ) -> None:
        """Initialize a new NaturalnessDataset dataset instance."""

        assert split in ["train", "validation", "test"]
        self.root = root
        self.split = split
        self.crop_size = crop_size
        self.larger_image_size = 1920
        self.transforms = transforms
        self.seed = seed
        self.return_modality = return_modality
        print("Using the Modality: ", self.return_modality)

        split_dataframe = pd.read_csv(split_file)
        self.ids = split_dataframe[split].dropna().values.tolist()
        self.ids = [int(i) for i in self.ids]

        self.rng = (
            np.random.default_rng(int(seed))
            if seed is not None
            else np.random.default_rng()
        )

    def __getfilepath__(self, index):
        filename = self.ids[index]
        full_filename = os.path.join(self.root, "NI", f"{filename}.tif")
        return full_filename

    def __getitem__(self, index):
        filename = self.ids[index]
        season = "s2_temporal_subset"
        full_filename = os.path.join(self.root, str(season), f"{filename}.tif")

        patch_size = self.crop_size

        x_patch = self.rng.integers(0, self.larger_image_size - patch_size)
        y_patch = self.rng.integers(0, self.larger_image_size - patch_size)

        mask = self._load_raster(filename, "NI", patch_size, x_patch, y_patch)
        mask = mask / 100

        image = None

        if self.return_modality == "s2" or self.return_modality == "all":
            s2 = (
                self._load_raster(filename, season, patch_size, x_patch, y_patch)
                / 10000
            )  # Load and normalize S2
            image = s2 if image is None else torch.cat([image, s2], dim=0)

        if self.return_modality == "s1" or self.return_modality == "all":
            s1 = self._load_raster(filename, "S1", patch_size, x_patch, y_patch)
            s1[0, :, :] = self.normalize(s1[0, :, :], VV_STATS)  # Normalize VV
            s1[1, :, :] = self.normalize(s1[1, :, :], VH_STATS)  # Normalize VH
            image = s1 if image is None else torch.cat([image, s1], dim=0)

        if self.return_modality == "esa_wc" or self.return_modality == "all":
            esa_wc = self._load_raster(filename, "ESA_WC", patch_size, x_patch, y_patch)
            esa_wc = self.integer_encode(esa_wc)  # Apply categorical encoding
            image = esa_wc if image is None else torch.cat([image, esa_wc], dim=0)

        if self.return_modality == "viirs" or self.return_modality == "all":
            viirs = self._load_raster(filename, "VIIRS", patch_size, x_patch, y_patch)
            viirs = self.normalize(viirs, AVG_RAD_STATS)  # Normalize VIIRS
            image = viirs if image is None else torch.cat([image, viirs], dim=0)

        sample: dict[str, Tensor] = {"image": image, "mask": mask}
        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample, filename

    def __len__(self) -> int:
        """Return the number of data points in the dataset.
        Returns:
            length of the dataset
        """
        return len(self.ids)

    def _load_raster(
        self,
        filename: int,
        source: str,
        crop_size: int,
        x=None,
        y=None,
    ) -> torch.Tensor:
        filepath = os.path.join(self.root, source, f"{filename}.tif")
        with rasterio.open(filepath) as f:
            window = rasterio.windows.Window(x, y, crop_size, crop_size)
            raw_array = f.read(window=window)
            array = np.stack(raw_array, axis=0)
            if array.dtype == np.uint16:
                array = array.astype(np.int32)
            tensor = torch.from_numpy(array).float()
            return tensor

    @staticmethod
    def normalize(tensor, stats):
        return (tensor - stats["mean"]) / stats["std"]

    @staticmethod
    def integer_encode(array):
        min_val = 10.0
        max_val = 100.0
        normalized_array = (array - min_val) / (max_val - min_val)
        return normalized_array.clone().detach()
