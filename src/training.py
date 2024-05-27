# -*- coding: utf-8 -*-
"""
Created on Fri May 24 17:17:58 2024

@author: Ammar Abdurrauf
"""

import os
import numpy as np
import tensorflow as tf
from src.utils import AppelDataset
from src.unet import UNet
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau

# set seed for deterministic random
os.environ["PYTHONHASHSEED"] = str(42)
np.random.seed(42)
tf.random.set_seed(42)

# hyperparameters
batch_size = 4
lr = 0.0001
epochs = 200
height = width = 256

training_no = "002"

# directories
dataset_dir = "D:/Projects/rotten-apple-unet/dataset/"
model_dir = "D:/Projects/rotten-apple-unet/models/"
log_file = "D:/Projects/rotten-apple-unet/log/log-" + training_no + ".csv"

# fetch the dataset
apple = AppelDataset(height, width, path=dataset_dir)
# train, validation = apple.get_dataset()
train_batch, valid_batch = apple.get_dataset(batch_size)

# display the dataset
# apple.display_image_mask(train)
apple.display_image_mask_batch(train_batch)

with tf.device('/device:GPU:0'):
    # define the model
    unet = UNet(height, width)
    unet.model_summary()
    
    # compile the model
    unet.compile_model()
    
    # set the callbacks (save the best and last models and log file)
    callbacks = [
            ModelCheckpoint(model_dir + 'RAS_model_best_' + training_no + '.h5',
            save_best_only=True,
            verbose=1),
            ModelCheckpoint(model_dir + 'RAS_model_last_' + training_no + '.h5',
            save_best_only=False,
            verbose=1),
            # ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5),
            CSVLogger(log_file)
        ]
    
    # train the model
    unet.model.fit(train_batch, 
              validation_data=valid_batch, 
              epochs=epochs,
              callbacks=callbacks)

# shoe the loss and accuracy during training
unet.show_history(log_file)

# test the model
validation = unet.model.predict(valid_batch, verbose=1)
apple.ground_truth_and_predicted_mask(valid_batch, validation)



""" delete after this - display dataset """
import matplotlib.pyplot as plt

images, masks = apple.get_images_and_masks()

start_index=200
for i in range(1, 5):
    plt.subplot(2,5,i)
    plt.imshow(images[start_index], cmap='gray')
    
    plt.subplot(2,5,i+5)
    plt.imshow(masks[start_index], cmap='gray')
    start_index += 1
    plt.show