# -*- coding: utf-8 -*-
"""
Created on Fri May 24 17:22:50 2024

@author: Ammar Abdurrauf
"""

import os
import cv2
import random
import numpy as np
from glob import glob
import tensorflow as tf
import matplotlib.pyplot as plt

class AppelDataset:
    def __init__(self, height, width, path="dataset"):
        self.height = height
        self.width = width
        self.path = path
            
    def load_data(self):
        img_files = sorted(glob(os.path.join(self.path, "images", "*")))
        mask_files = sorted(glob(os.path.join(self.path, "masks", "*")))
        
        combined_list = list(zip(img_files, mask_files))
        random.seed(42)
        random.shuffle(combined_list)
        img_files, mask_files = zip(*combined_list)
        
        return list(img_files), list(mask_files)
    
    def read_image(self, path):
        path = path.decode()
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.height, self.width), interpolation=cv2.INTER_AREA)
        img = img/255.0
        return img
    
    def read_mask(self, path):
        path = path.decode()
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.height, self.width), interpolation=cv2.INTER_AREA)
        mask = mask/255.0
        mask = np.where(mask >= 0.5, 1.0, 0.0)
        mask = np.expand_dims(mask, axis=-1)
        return mask
    
    def read_and_resize(self, img, mask):
        def _load(img, mask):
            img = self.read_image(img)
            mask = self.read_mask(mask)
            return img, mask
        
        img, mask = tf.numpy_function(_load, [img, mask], [tf.float64, tf.float64])
        img.set_shape([self.height, self.width, 3])
        mask.set_shape([self.height, self.width, 1])
        return img, mask
    
    def get_dataset(self, batch_size=None):
        img_files, mask_files = self.load_data()       
        dataset = tf.data.Dataset.from_tensor_slices((img_files, mask_files))                
        dataset = dataset.map(self.read_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
        
        if batch_size != None and batch_size > 0:
            dataset = dataset.batch(batch_size)
        
        train_dataset = dataset.take(int(dataset.cardinality().numpy()*0.9))
        val_dataset = dataset.skip(int(dataset.cardinality().numpy()*0.9))
        
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
        
        return train_dataset, val_dataset
    
    def get_images_and_masks(self):
        images = []
        masks = []
        img_files, mask_files = self.load_data()
        
        for i in range(len(img_files)):
            img = cv2.imread(img_files[i], cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img/255.0
            img = cv2.resize(img, (self.height, self.height))
            images.append(img)
            
            mask = cv2.imread(mask_files[i], cv2.IMREAD_GRAYSCALE)
            mask = mask/255.0
            mask = cv2.resize(mask, (self.height, self.height))
            mask = np.where(mask >= 0.5, 1.0, 0.0)
            masks.append(mask)
            
        return images, masks
    
    
    def display_image_mask(self, dataset):
        images = list(map(lambda x: x[0], dataset))
        masks = list(map(lambda x: x[1], dataset))
        
        i = random.randint(0, len(images)-1)
        plt.subplot(1,2,1)
        plt.imshow(images[i])
        plt.title('image')
        plt.grid(False)
        
        plt.subplot(1,2,2)
        plt.imshow(masks[i])
        plt.title('mask')
        plt.grid(False)
        
        plt.show()
        
    def display_image_mask_batch(self, dataset):
        images = list(map(lambda x: x[0], dataset))
        masks = list(map(lambda x: x[1], dataset))
        
        i = random.randint(0, len(images)-1)
        j = random.randint(0, len(images[i])-1)
        plt.subplot(1,2,1)
        plt.imshow(images[i][j])
        plt.title('image')
        plt.grid(False)
        
        plt.subplot(1,2,2)
        plt.imshow(masks[i][j])
        plt.title('mask')
        plt.grid(False)
        
        plt.show()
        
    def ground_truth_and_predicted_mask(self, dataset, val_result):
        
        img = list(map(lambda x: x[0], dataset))
        gt= list(map(lambda x: x[1], dataset))
        batch_size = len(img[0])
        
        i = random.randint(0, len(img)-1)
        j = random.randint(0, len(img[i])-1)
        
        # print(i, j)
        
        plt.subplot(2,2,1)
        plt.imshow(img[i][j])
        plt.title('image')
        plt.grid(False)
        
        plt.subplot(2,2,2)
        plt.imshow(gt[i][j])
        plt.title('true mask')
        plt.grid(False)
        
        plt.subplot(2,2,3)
        plt.imshow(val_result[i*batch_size+j])
        plt.title('predicted mask')
        plt.grid(False)
        
        thres = [[1 if x >= 0.5 else 0 for x in m] for m in val_result[i*batch_size+j]]
        plt.subplot(2,2,4)
        plt.imshow(thres, cmap='gray')
        plt.title('pred. mask > 0.5 tresh')
        plt.rcParams.update({'font.size': 6})
        plt.grid(False)
        
        plt.show()
