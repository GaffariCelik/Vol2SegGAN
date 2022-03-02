
from tensorflow.keras.models import Model,load_model,model_from_json
from tensorflow.keras.layers import Input, Conv3D,UpSampling3D, Conv3DTranspose, Dropout,Activation, ReLU, LeakyReLU, Concatenate,BatchNormalization
from tensorflow.keras.optimizers import Adam
import tensorflow.compat.v1 as tf

import tensorflow_addons as tfa
from tensorflow_addons.layers import InstanceNormalization

from random import randint
import os
import numpy as np
from utils2 import  get_dsc
import config

#from attention import PAM, CAM
FLAGS = tf.app.flags.FLAGS


class Vol2SegGAN():
    def __init__(self, img_shape, seg_shape, Nfilter_start=64, depth=4):
        self.img_shape = img_shape
        self.seg_shape = seg_shape
        self.Nfilter_start = Nfilter_start
        self.depth = depth
        
    def Generator(self):

        inputs = Input(self.img_shape, name='input_image')     

        def encoder_step(layer, Nf, inorm=True):            
            x = Conv3D(Nf, kernel_size=3, strides=2, kernel_initializer='he_normal', padding='same')(layer)
            x = InstanceNormalization()(x)
            x = LeakyReLU()(x)
            x=Dropout(0.2)(x)

            x = Conv3D(Nf*2, kernel_size=3,kernel_initializer='he_normal', padding='same')(x)
            x = InstanceNormalization()(x)
            x = LeakyReLU()(x)
            x=Dropout(0.2)(x)            
            
            return x
        def Conv3d_BN(x, nb_filter, kernel_size, strides=1, padding='same', use_activation=True):
            x = Conv3D(nb_filter, kernel_size, padding=padding, strides=strides, kernel_initializer='he_normal')(x)
            x = BatchNormalization(axis=3)(x)
            if use_activation:
                x = Activation('relu')(x)
                return x
            else:
                return x

        def Conv3d(layer):
            #x = PAM()(layer)
            x = Conv3D(256, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(layer)
            x = BatchNormalization(axis=3)(x)
            x = Activation('relu')(x)
            x = Dropout(0.5)(x)
            x = Conv3D(256, 3, padding='same', use_bias=False, kernel_initializer='he_normal')(x)

            return x


        def future_translate(layer, Nf):

            x_atr = Conv3D(Nf, kernel_size=3, strides=2, kernel_initializer='he_normal', padding='same')(layer)           
            x_atr = InstanceNormalization()(x_atr)
            x_atr = LeakyReLU()(x_atr)

            y1 = Conv3D(50, kernel_size=3, strides=1, dilation_rate=1,  padding='same')(x_atr)
            y2 = Conv3D(50, kernel_size=3, strides=1, dilation_rate=3,  padding='same')(x_atr)
            y3 = Conv3D(50, kernel_size=3, strides=1, dilation_rate=5,  padding='same')(x_atr)
            y3=UpSampling3D(size=1)(y3)
            y2_3=Concatenate()([y2,y3])
            y2_3=UpSampling3D(size=1)(y2_3)
            y=Concatenate()([y1,y2_3])
            y=Conv3d(y)


            return y

        
        def decoder_step(layer, layer_to_concatenate, Nf):
            x = Conv3DTranspose(Nf, kernel_size=5, strides=2, padding='same', kernel_initializer='he_normal')(layer)
            x = InstanceNormalization()(x)
            x = LeakyReLU()(x)
            x = Concatenate()([x, layer_to_concatenate])
            x = Dropout(0.2)(x)

            x = Conv3D(Nf, kernel_size=3,kernel_initializer='he_normal', padding='same')(x)
            x = InstanceNormalization()(x)
            x = LeakyReLU()(x)
            x=Dropout(0.2)(x)             
            return x

          

        layers_to_concatenate = []
        x = inputs

        # encoder
        for d in range(self.depth-1):
            x = encoder_step(x, self.Nfilter_start*np.power(2,d))
            layers_to_concatenate.append(x)
        
        # future_translate
        x = future_translate(x, self.Nfilter_start*np.power(2,self.depth-1))

        # decoder
        for d in  range(self.depth-2, -1, -1): 
            x = decoder_step(x, layers_to_concatenate.pop(), self.Nfilter_start*np.power(2,d))
        # classifier
        last = Conv3DTranspose(4, kernel_size=4, strides=2, padding='same', kernel_initializer='he_normal', activation='softmax', name='output_generator')(x)
        # Create model
        return Model(inputs=inputs, outputs=last, name='Generator')

    def Discriminator(self):

        inputs = Input(self.img_shape, name='input_image')
        targets = Input(self.seg_shape, name='target_image')

        def encoder_step(layer, Nf, inorm=True):
            x = Conv3D(Nf, kernel_size=5, strides=2, kernel_initializer='he_normal', padding='same')(layer)
            if inorm:
                x = InstanceNormalization()(x)
            x = LeakyReLU()(x)
            x = Dropout(0.2)(x)
            return x

        x = Concatenate()([inputs, targets])

        for d in range(self.depth):
            if d==0:
                x = encoder_step(x, self.Nfilter_start*np.power(2,d), False)
            else:
                x = encoder_step(x, self.Nfilter_start*np.power(2,d))


        last = tf.keras.layers.Conv3D(1, 5, strides=1, padding='same', kernel_initializer='he_normal', name='output_discriminator')(x) 

        return Model(inputs=[targets, inputs], outputs=last, name='Discriminator')
    
