#!/usr/bin/env python
# coding: utf-8

import argparse
import glob
import os
import shutil
from shutil import copy, move

import numpy as np
import pandas as pd
import PIL.Image as Image
import tqdm
from PIL import Image as Image
from skimage import io

import cv2
import faiss
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
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


class TwoLevelClustering(object):
    def __init__(self,
                 images_path,
                 output_path,
                 bucket_path="",
                 batch_size=48,
                 cluster_levels=[8, 8],
                 image_size=(300, 300),
                 arch="BIT"):
        self.output_path = output_path
        self.bucket_path = bucket_path
        self.image_size = image_size
        self.cluster_levels = cluster_levels
        self.batch_size = batch_size

        self.ds, self.dl = self._get_dataset_dataloader(images_path)
        self.model = self._get_model()
        self.df = self.feature_extraction()
        self.clustering()
        self.move()

    def _get_dir_images(self, image_path):
        images_list = glob.glob("{}/*.jpg".format(image_path))
        return images_list

    # def _select_arch(self,arch):
    #     # if arch == "MobileNet":
    #     #     module = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4",)

    #     # else:
    #     #     module = hub.KerasLayer("https://tfhub.dev/google/bit/m-r50x1/1")
    #     # return module

    def _get_model(self):
        model = models.resnext50_32x4d(pretrained=True)
        model = torch.nn.Sequential(*list(model.children())[:-1])
        model = model.cuda()
        model.eval()
        return model

    def _get_dataset_dataloader(self, images_path):
        transform = trans.Compose([
            trans.ToPILImage(),
            trans.Resize((300, 300)),
            trans.ToTensor(),
            trans.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225))
        ])
        ds = ClusterDataset(images_path, transform)
        loader = DataLoader(ds, batch_size=self.batch_size,
                            pin_memory=True, num_workers=4)
        return ds, loader  # , class_num

    def feature_extraction(self):
        # for image_path in tqdm.tqdm(self.images_list):
        #     img = Image.open(image_path).resize((300,300))
        #     img = np.array(img)/255.0
        #     img = np.reshape(img,(1,300,300,3))
        #     feat = self.model(img)
        #     feat_all.append(feat.numpy().squeeze())
        #     path_all.append(image_path)

        # df = pd.DataFrame()
        # df["features"] = feat_all
        # df["img_path"] = path_all

        # return df

        emb = []
        names = []
        # with self.model.zero_grad():
        for idx, (imgs, paths) in enumerate(tqdm.tqdm(self.dl)):
            imgs = imgs.to('cuda')
            embeddings = self.model(imgs)
            embeddings = embeddings.squeeze().cpu().detach().numpy()
            # paths = paths.cpu().detach().numpy().tolist()

            for idy, embedding in enumerate(embeddings):
                # emb[batch_size*idx+idy,:] = embedding
                emb.append(embedding)
                names.append(paths[idy])

        df = pd.DataFrame()
        df["features"] = emb
        df["img_path"] = names
        return df

    def clustering(self):
        
        feat = np.array(self.df.features.values)
        feat = np.stack(feat, axis=0)

        ncentroids = self.cluster_levels[0]
        niter = 20
        verbose = True
        d = feat.shape[1]
        kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
        kmeans.train(feat)
        D, I = kmeans.index.search(feat, 1)
        self.df["cluster1"] = I.T[0]

        df_group = self.df.groupby(self.df.cluster1)
        
        df_list = []
        
        for idx, df in df_group:
            a = np.array(df.features.values)
            a = np.stack(a, axis=0)
            ncentroids = 8 
            niter = 20
            verbose = True
            # print(a)
            d = a.shape[1]
            kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
            kmeans.train(a)
            D, I = kmeans.index.search(a, 1)
            df.loc[:, "cluster2"] = I.T[0]
            df_list.append(df)
        
        self.df = pd.concat(df_list)
        self.df["combined_cluster"] = self.df["cluster1"].astype(str) \
                       + "_" \
                       + self.df["cluster2"].astype(str)
        

    def _chunk(self,lst,chunk_size=16):
        chunk_lst = []        
        tail = round((len(lst)/chunk_size - len(lst)//chunk_size) * chunk_size)

        for i in range(len(lst)//chunk_size):
            start = i*chunk_size
            end = (i+1)*chunk_size
            chunk_lst.append(lst[start:end])

        if len(lst[end:end+tail]):
            chunk_lst.append(lst[end:end+tail])
        
        return chunk_lst
        
    def move(self):    
        
        paths = self.df.img_path.values
        cluster = self.df.combined_cluster.values
        
        try:
            shutil.rmtree(self.output_path)
        except Exception as E:
            print(E)
        
        os.mkdir(self.output_path)

        for idx in self.df["combined_cluster"].unique():
            os.mkdir(os.path.join(self.output_path, str(idx)))

        for idx, i in enumerate(cluster):
            copy_location = os.path.join(self.output_path, str(i))
            copy(paths[idx], copy_location)
      


if __name__ == "__main__":
    class Args():
        input_path = ""
        output_path = ""

    args = Args()
    cl = TwoLevelClustering(images_path=args.input_path,
                            output_path=args.output_path)
