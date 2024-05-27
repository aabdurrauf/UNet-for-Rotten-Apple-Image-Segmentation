# -*- coding: utf-8 -*-
"""
Created on Fri May 24 16:30:33 2024

@author: Ammar Abdurrauf
"""

from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dropout, Conv2DTranspose, Concatenate, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class UNet:
    def __init__(self, height=400, width=400, with_batch=False):
        input_shape = (height, width, 3)
        if with_batch:
            self.model = self.create_unet_model_with_batch(input_shape)
        else:
            self.model = self.create_unet_model(input_shape)
        
    def convolution_block(self, inputs, num_filters, num_kernels=3, pad='same', kernel_init='he_normal', activation='relu', drop_val=0.0):
        x = Conv2D(num_filters, num_kernels, padding=pad, kernel_initializer=kernel_init)(inputs)
        x = Activation(activation)(x)
        x = Dropout(drop_val)(x)
        
        x = Conv2D(num_filters, num_kernels, padding=pad, kernel_initializer=kernel_init)(inputs)
        x = Activation(activation)(x)
        
        return x
    
    def convolution_block_with_batch(self, inputs, num_filters, num_kernels=3, pad='same', kernel_init='he_normal', activation='relu'):
        x = Conv2D(num_filters, num_kernels, padding=pad, kernel_initializer=kernel_init)(inputs)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        
        x = Conv2D(num_filters, num_kernels, padding=pad, kernel_initializer=kernel_init)(inputs)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
        
        return x
    
    def encoder_block(self, inputs, num_filters, with_batch):
        if with_batch:
            x = self.convolution_block_with_batch(inputs, num_filters)
        else:
            x = self.convolution_block(inputs, num_filters, drop_val=0.5)
        m = MaxPool2D(2)(x)
        return x, m
    
    def decoder_block(self, inputs, skip, num_filters, pad='same'):
        x = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding=pad)(inputs)
        x = Concatenate()([x, skip])
        x = self.convolution_block(x, num_filters)
        return x
        
    def create_unet_model(self, input_shape):
        inputs = Input(input_shape)
        
        # Encoder Part
        s1, e1 = self.encoder_block(inputs, 64, False)
        s2, e2 = self.encoder_block(e1, 128, False)
        s3, e3 = self.encoder_block(e2, 256, False)
        s4, e4 = self.encoder_block(e3, 512, False)
        
        c1 = self.convolution_block(e4, 1024)
        
        # Decoder Part
        d1 = self.decoder_block(c1, s4, 512)
        d2 = self.decoder_block(d1, s3, 256)
        d3 = self.decoder_block(d2, s2, 128)
        d4 = self.decoder_block(d3, s1, 64)
        
        outputs = Conv2D(1, 1, padding='same', activation='sigmoid')(d4)
        model = Model(inputs, outputs, name='UNet')
        return model
    
    def create_unet_model_with_batch(self, input_shape):
        inputs = Input(input_shape)
        
        # Encoder Part
        s1, e1 = self.encoder_block(inputs, 64, True)
        s2, e2 = self.encoder_block(e1, 128, True)
        s3, e3 = self.encoder_block(e2, 256, True)
        s4, e4 = self.encoder_block(e3, 512, True)
        
        c1 = self.convolution_block(e4, 1024)
        
        # Decoder Part
        d1 = self.decoder_block(c1, s4, 512)
        d2 = self.decoder_block(d1, s3, 256)
        d3 = self.decoder_block(d2, s2, 128)
        d4 = self.decoder_block(d3, s1, 64)
        
        outputs = Conv2D(1, 1, padding='same', activation='sigmoid')(d4)
        model = Model(inputs, outputs, name='UNet')
        return model
    
    def compile_model(self, 
                      opt=Adam(learning_rate=0.0001), 
                      loss_function='binary_crossentropy'):
        
        # self.model.compile(optimizer=opt, loss=loss_function, 
        #                    metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])
        self.model.compile(optimizer=opt, loss=loss_function, metrics=['accuracy'])
        
    def model_summary(self):
        self.model.summary()
        
    def show_history(self, log_file):
        history = pd.read_csv(log_file)
        sns.set(style="whitegrid")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        sns.lineplot(x="epoch", y="accuracy", data=history, ax=ax1, label="Training Accuracy")
        sns.lineplot(x="epoch", y="val_accuracy", data=history, ax=ax1, label="Validation Accuracy")
        
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Model Accuracy")
        ax1.legend()
        
        sns.lineplot(x="epoch", y="loss", data=history, ax=ax2, label="Training Loss")
        sns.lineplot(x="epoch", y="val_loss", data=history, ax=ax2, label="Validation Loss")
        
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Loss")
        ax2.set_title("Model Loss")
        ax2.legend()
        
        plt.tight_layout()
        plt.show()