import glob

import numpy as np
import PIL.Image as Image
import torchvision.datasets as datasets
# from PIL import Image as Image
from skimage import io
from torch.utils.data import Dataset
from torchvision import transforms as trans


class ClusterDataset(Dataset,):
    def __init__(self,
                 image_path,
                 transforms=None):

        self.image_path = image_path
        self.samples = glob.glob("{}/*.jpg".format(image_path))
        # self.samples = self.samples[0:5000]
        self.transform = transforms

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image = io.imread(self.samples[idx])
        # img_name = self.samples[idx].split("/")[-1]

        if self.transform:
            image = self.transform(image)
        return image, self.samples[idx]
