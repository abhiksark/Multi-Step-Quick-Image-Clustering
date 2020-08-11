#!/usr/bin/env python
# coding: utf-8

import os
import glob
import shutil
from shutil import copy

import numpy as np
import PIL.Image as Image
import tqdm
import pandas as pd 

import faiss
import tensorflow as tf
import tensorflow_hub as hub


class TwoLevelClustering(object):
    def __init__(self,
                 image_path,
                 output_path,
                 bucket_path="", 
                 cluster_levels=[12,5],
                 image_size=(300,300),
                 arch="BIT"):
        self.output_path = output_path
        self.bucket_path = bucket_path
        self.image_size = image_size
        self.cluster_levels = cluster_levels
        
        self.images_list = self._get_dir_images(image_path)
        self.model = self._select_arch(arch)
        self.df  = self.feature_extraction()

    def _get_dir_images(self,image_path):
        images_list = glob.glob("{}/*.jpg".format(image_path))
        return images_list

    def _select_arch(self,arch):
        if arch == "MobileNet":
            module = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4",)

        else:
            module = hub.KerasLayer("https://tfhub.dev/google/bit/m-r50x1/1")
        return module

    def feature_extraction(self):
        feat_all = []
        path_all = []
        for image_path in tqdm.tqdm(self.images_list):  
            img = Image.open(image_path).resize((300,300))
            img = np.array(img)/255.0
            img = np.reshape(img,(1,300,300,3))
            feat = self.model(img)
            feat_all.append(feat.numpy().squeeze())
            path_all.append(image_path)

        df = pd.DataFrame()
        df["features"] = feat_all
        df["img_path"] = path_all

        return df
    
    def clustering(self):
        feat = np.array(self.df.features.values)
        # print(feat)
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
            ncentroids = 2
            niter = 20
            verbose = True
            # print(a)
            d = a.shape[1]
            kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
            kmeans.train(a)
            D, I = kmeans.index.search(a, 1)
            df.loc[:,"cluster2"] = I.T[0]
            df_list.append(df)
        self.df = pd.concat(df_list)
        self.df["c"]  = self.df["cluster1"].astype(str) + "_" + self.df["cluster2"].astype(str)
        paths = self.df.img_path.values
        cluster = self.df.c.values
        try:
            shutil.rmtree(self.output_path)
        except Exception as E:
            print(E)
        os.mkdir(self.output_path)
        
        for idx in self.df["c"].unique():
            os.mkdir(os.path.join(self.output_path,str(idx)))

        for idx,i in enumerate(cluster):
            copy_location = os.path.join(self.output_path,str(i))
            print(copy_location)
            copy(paths[idx],copy_location)
        

if __name__ == "__main__":
    class Args():
        input_path = "data/bi_1"
        output_path = "cluster1"

    args = Args()
    cl = TwoLevelClustering(image_path = args.input_path, output_path = args.output_path)
    cl.clustering()

