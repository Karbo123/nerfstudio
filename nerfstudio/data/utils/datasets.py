# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Some dataset code.
"""
from __future__ import annotations

from typing import Dict

import cv2
import os.path as osp

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchtyping import TensorType

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs


class InputDataset(Dataset):
    """Dataset that returns images.

    Args:
        dataparser_outputs: description of where and how to read input images.
    """

    def __init__(self, dataparser_outputs: DataparserOutputs):
        super().__init__()
        self.dataparser_outputs = dataparser_outputs

    def __len__(self):
        return len(self.dataparser_outputs.image_filenames)

    def get_numpy_image(self, image_idx: int) -> npt.NDArray[np.uint8]:
        """Returns the image of shape (H, W, 3 or 4).

        Args:
            image_idx: The image index in the dataset.
        """
        image_filename = self.dataparser_outputs.image_filenames[image_idx]
        pil_image = Image.open(image_filename)
        image = np.array(pil_image, dtype="uint8")  # shape is (h, w, 3 or 4)
        assert len(image.shape) == 3
        assert image.dtype == np.uint8
        assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is in correct."
        return image

    def get_image(self, image_idx: int) -> TensorType["image_height", "image_width", "num_channels"]:
        """Returns a 3 channel image.

        Args:
            image_idx: The image index in the dataset.
        """
        image = torch.from_numpy(self.get_numpy_image(image_idx).astype("float32") / 255.0)
        if self.dataparser_outputs.alpha_color is not None and image.shape[-1] == 4:
            assert image.shape[-1] == 4
            image = image[:, :, :3] * image[:, :, -1:] + self.dataparser_outputs.alpha_color * (1.0 - image[:, :, -1:])
        else:
            image = image[:, :, :3]
        return image

    def get_data(self, image_idx) -> Dict:
        """Returns the ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
        """
        image = self.get_image(image_idx)
        data = {"image_idx": image_idx}
        data["image"] = image
        for _, data_func_dict in self.dataparser_outputs.additional_inputs.items():
            assert "func" in data_func_dict, "Missing function to process data: specify `func` in `additional_inputs`"
            func = data_func_dict["func"]
            assert "kwargs" in data_func_dict, "No data to process: specify `kwargs` in `additional_inputs`"
            data.update(func(image_idx, **data_func_dict["kwargs"]))
        return data

    def get_depth_data(self, image_idx):
        """ try load depth image from disk, if not exist, return None
        """
        depth_scale = self.dataparser_outputs.depth_scale
        image_filename = self.dataparser_outputs.image_filenames[image_idx]
        depth_filename = str(image_filename).replace(".png", "_depth.png")
        if depth_scale > 0 and osp.exists(depth_filename):
            depth = cv2.imread(depth_filename, cv2.IMREAD_UNCHANGED)
            assert len(depth.shape) == 2 # (H, W)
            assert depth.dtype == np.uint16
            depth = depth.astype("float32") * depth_scale
            depth = torch.from_numpy(depth)
            return depth
    
    def __getitem__(self, image_idx):
        data = self.get_data(image_idx)
        # try to load depth data
        depth = self.get_depth_data(image_idx)
        if depth is not None:
            data["depth"] = depth
            data["depth_min"] = self.dataparser_outputs.depth_min
        return data
