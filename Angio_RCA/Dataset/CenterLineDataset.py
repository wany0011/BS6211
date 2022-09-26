# -*- coding: utf-8 -*-

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class CenterLineDataset(Dataset):
    def __init__(self, input_images, central_lines, image_transform=None):
        self.input_images = []
        # self.annotations = []
        self.central_lines = []
        
        # print(len(input_images))
        for idx in range(len(input_images)):
            self.input_images.append(image_transform.fit_transform(input_images[idx]))
            # self.central_lines.append(image_transform.fit_transform(central_lines[idx]))
            self.central_lines.append(central_lines[idx]/255.0)

        """
        for idx in range(len(input_images)):
            img = np.zeros((512, 512), dtype=np.uint8)
            coordinates = (annotations[idx] * 512).astype(np.uint8)
            for j in range(coordinates.shape[0]):
                img[coordinates[j][0], coordinates[j][1]] = 255
            self.output_images.append(img)
        """

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        return self.input_images[idx], self.central_lines[idx]
