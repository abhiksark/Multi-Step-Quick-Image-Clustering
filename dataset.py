import argparse
import glob
import os
import shutil
from shutil import copy, move

import cv2
import faiss
import numpy as np
import pandas as pd
import PIL.Image as Image
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import tqdm
from PIL import Image as Image
from skimage import io
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as trans
from torchvision.datasets import ImageFolder

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

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
