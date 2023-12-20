import datetime
import tensorflow as tf
import numpy as np
import os
import pickle

from utils.DataLoader import data_loader
from utils.Generator0 import DataGenerator, feature_extract_cqt, evalEER,  evalScore, evalEER_f, evalEER_f2, gen_fname
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import add,concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, maximum, DepthwiseConv2D, AveragePooling2D, UpSampling2D
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, SpatialDropout2D
from tensorflow.keras.layers import Convolution2D, GlobalAveragePooling2D, MaxPool2D, ZeroPadding2D
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from tensorflow.keras.activations import relu, softmax, swish
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint




dropout_rate = 0.5

def sigmoidal_decay(e, start=0, end=100, lr_start=1e-3, lr_end=1e-5):
    if e < start:
        return lr_start    
    if e > end:
        return lr_end    
    middle = (start + end) / 2
    s = lambda x: 1 / (1 + np.exp(-x))    
    return s(13 * (-e + middle) / np.abs(end - start)) * np.abs(lr_start - lr_end) + lr_end

def SubSpectralNorm(S, A = True, eps=1e-5):
    # S : number of sub-bands
    # A : 'Sub' Transform type if True, 
    #     'All' Transform type if False.          <- cannot be implemented yet.
    # Can be applied only for image inputs.
    
    
    def f(x):
        if S == 1:
            y = BatchNormalization()(x)
            
        else:
            F = x.shape[1]             # number of frequencies of the input.
            if F % S == 0:
                Q = F // S             # length of each sub-band.
                subs = []
                for i in range(S):     
                    subs.append(x[:, i*Q:(i+1)*Q, :,:])
                    
                for i in range(S):
                    subs[i] = BatchNormalization()(subs[i])
                    
            else:
                Q = F // S
                subs = []
                for i in range(S-1):
                    subs.append(x[:,i*Q:(i+1)*Q,:,:])
                    
                subs.append(x[:,(S-1)*Q:,:,:])      # the shape of x and y must be the same.
                
                for i in range(S):
                    subs[i] = BatchNormalization()(subs[i])
                    
            
            y = tf.concat(subs, axis=1)
            
        return y
    
    return f


def TF_ResMax_seq(out_channels = False, strides=(1,1), dilation_rate=(1,1),
                  freq_kernel_size=(3,1), temporal_kernel_size=(1,3), dropout_rate=0.3):
    
    def f(x):
        if out_channels:
            y = Conv2D(filters=out_channels, kernel_size=(1,1), strides=strides,
                       activation=None, kernel_initializer='he_uniform')(x)
            y = BatchNormalization()(y)
            y = relu(y)
            
            z = DepthwiseConv2D(kernel_size=freq_kernel_size, strides=(1,1), padding='same',
                                activation=None, kernel_initializer='he_uniform')(y)
            z = SubSpectralNorm(S=2)(z)
            
        else :
            z = DepthwiseConv2D(kernel_size=freq_kernel_size, strides=(1,1), padding='same',
                                activation=None, kernel_initializer='he_uniform')(x)
            z = SubSpectralNorm(S=2)(z)
            
        z = relu(z)
        z = DepthwiseConv2D(kernel_size=temporal_kernel_size, strides=(1,1), padding='same',
                                activation=None, kernel_initializer='he_uniform')(z)
        z = SubSpectralNorm(S=2)(z)
            #        y = relu(y)
        z = swish(z)
            
        if out_channels:
            z = Conv2D(filters=y.shape[3], kernel_size=(1,1), strides=(1,1),
                    activation='relu', kernel_initializer='he_uniform')(z)
        else :
            z = Conv2D(filters=x.shape[3], kernel_size=(1,1), strides=(1,1),
                       activation='relu', kernel_initializer='he_uniform')(z)
                
        z = SpatialDropout2D(dropout_rate)(z)
        ############################
            
        if out_channels:
            return add([y, z])
        else:
            return add([x, z])
            
    return f


    
def BC_ResMax(out_channels = False, strides=(1,1), dilation_rate=(1,1), 
              freq_kernel_size=(3,1), temporal_kernel_size=(1,3), dropout_rate=0.1):

    def f(x):
        if out_channels:
            y = Conv2D(filters=out_channels, kernel_size=(1,1), strides=strides, 
                       activation=None, kernel_initializer='he_uniform')(x)
            y = BatchNormalization()(y)
            y = relu(y)
            
            y = DepthwiseConv2D(kernel_size=freq_kernel_size, strides=(1,1), padding='same',
                                 activation=None, kernel_initializer='he_uniform')(y)
#            y2 = DepthwiseConv2D(kernel_size=freq_kernel_size, strides=(1,1), padding='same',
#                                 activation=None, kernel_initializer='he_uniform')(y)
#            y = tf.keras.layers.maximum([y1, y2])

        else :
            y = DepthwiseConv2D(kernel_size=freq_kernel_size, strides=(1,1), padding='same',
                                 activation=None, kernel_initializer='he_uniform')(x)
#            y2 = DepthwiseConv2D(kernel_size=freq_kernel_size, strides=(1,1), padding='same',
#                                 activation=None, kernel_initializer='he_uniform')(x)
#            y = tf.keras.layers.maximum([y1, y2])
            
        y = SubSpectralNorm(S=2)(y)                   # Should be changed to SSN
        y = relu(y)
        ############################
        
        z = AveragePooling2D(pool_size=(y.shape[1],1))(y)
        
        ########### f1 #############
        z = DepthwiseConv2D(kernel_size=temporal_kernel_size, strides=(1,1), dilation_rate=dilation_rate,
                            padding='same', activation=None, kernel_initializer='he_uniform')(z)
        z = BatchNormalization()(z)
        z = swish(z)

        z = Conv2D(filters=y.shape[3], kernel_size=(1,1), strides=(1,1),
                   activation='relu', kernel_initializer='he_uniform')(z)
        z = SpatialDropout2D(dropout_rate)(z)                  
        ############################
        
        
        ########### BC #############
        z = UpSampling2D(size=(y.shape[1],1), interpolation='nearest')(z)
        ############################
        
        if out_channels:
            return add([y, z])
        else: 
            return add([x, y, z])
        
    return f

def ResMax(n_output, k, l, upscale=False):
    def f(x):
        conv1_1 = Conv2D(filters = n_output, kernel_size =k, strides=(1, 1), padding='same', activation=None)(x)
        conv1_2 = Conv2D(filters = n_output, kernel_size =k, strides=(1, 1), padding='same', activation=None)(x)
        h = maximum([conv1_1, conv1_2])
        if l :
            conv1_1 = Conv2D(filters = n_output, kernel_size =k, strides=(1, 1), padding='same', activation=None)(h)
            conv1_2 = Conv2D(filters = n_output, kernel_size =k, strides=(1, 1), padding='same', activation=None)(h)
            h = maximum([conv1_1, conv1_2])
        if upscale:
            f = Conv2D(kernel_size=1, filters=n_output, strides=1, padding='same')(x)
        else:
            f = x
        return add([f, h])
    return f







def get_ResMax(input_shape):
        # Create a simple model.
    input_tensor = Input(input_shape)

    x = ResMax(16,3,1, upscale = True)(input_tensor)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)

    x = ResMax(16,5,1, upscale = False)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)

    x = ResMax(24,3,1, upscale = True)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
#    x = Dropout(dropout_rate)(x)
    
    x = ResMax(32,3,0, upscale = True)(x)
    x = BatchNormalization()(x)
    
    x = ResMax(32,3,0, upscale = False)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
#    x = Dropout(dropout_rate)(x)
    
    x = ResMax(48,3,0, upscale = True)(x)
    x = BatchNormalization()(x)
    
    x = ResMax(48,3,0, upscale = False)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
#    x = Dropout(dropout_rate)(x)
    
    x = ResMax(64,3,0, upscale = True)(x)
    x = BatchNormalization()(x)
#    x = Dropout(dropout_rate)(x)

    x = ResMax(64,3,0, upscale = False)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(units=2, activation = 'softmax')(x)
    model = Model(inputs=input_tensor, outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['AUC'])
    return model

def get_LCNN(input_shape):
    in1 = Input(shape = input_shape)
    c1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(in1)
    c2 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(in1)
    x = maximum([c1, c2])
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    c1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(x)
    c2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(x)
    x = maximum([c1, c2])
    x = BatchNormalization(axis=3, scale=False)(x)
    
    c1 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(x)
    c2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(x)
    x = maximum([c1, c2])
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = BatchNormalization(axis=3, scale=False)(x)

    c1 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(x)
    c2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(x)
    x = maximum([c1, c2])
    x = BatchNormalization(axis=3, scale=False)(x)

    c1 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(x)
    c2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(x)
    x = maximum([c1, c2])
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    c1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(x)
    c2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(x)
    x = maximum([c1, c2])
    x = BatchNormalization(axis=3, scale=False)(x)

    c1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(x)
    c2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(x)
    x = maximum([c1, c2])
    x = BatchNormalization(axis=3, scale=False)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    c1 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(x)
    c2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(x)
    x = maximum([c1, c2])
    x = BatchNormalization(axis=3, scale=False)(x)
    
    c1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(x)
    c2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(x)
    x = maximum([c1, c2])
    x = BatchNormalization(axis=3, scale=False)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    c1 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(x)
    c2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(x)
    x = maximum([c1, c2])
    x = BatchNormalization(axis=3, scale=False)(x)

    c1 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(x)
    c2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(x)
    x = maximum([c1, c2])
    x = BatchNormalization(axis=3, scale=False)(x)
    x = Dropout(.5)(x)

    x = GlobalAveragePooling2D(data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = Dropout(.5)(x)
    x = Dense(2, activation = 'softmax')(x)
    model = Model(inputs = in1, outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['AUC'])

    return model


def get_DDWSseq(input_shape):
    input_tensor = Input(input_shape)

    x1 = Conv2D(filters = 16, kernel_size = (3,3), activation = None)(input_tensor)
    x2 = Conv2D(filters = 16, kernel_size = (3,3), activation = None)(input_tensor)
    x = maximum([x1, x2])
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    x = TF_ResMax_seq(out_channels = False, strides=(1,1), dilation_rate=(1,1), 
                  freq_kernel_size=(3,1), temporal_kernel_size=(1,3), dropout_rate=0.1)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    x = TF_ResMax_seq(out_channels = 24, strides=(1,1), dilation_rate=(1,1), 
                  freq_kernel_size=(3,1), temporal_kernel_size=(1,3), dropout_rate=0.1)(x)
    x = TF_ResMax_seq(out_channels = False, strides=(1,1), dilation_rate=(1,1), 
                  freq_kernel_size=(3,1), temporal_kernel_size=(1,3), dropout_rate=0.1)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    x = TF_ResMax_seq(out_channels = 32, strides=(1,1), dilation_rate=(1,1), 
                  freq_kernel_size=(3,1), temporal_kernel_size=(1,3), dropout_rate=0.1)(x)
    x = TF_ResMax_seq(out_channels = False, strides=(1,1), dilation_rate=(1,1), 
                  freq_kernel_size=(3,1), temporal_kernel_size=(1,3), dropout_rate=0.1)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    x = TF_ResMax_seq(out_channels = 48, strides=(1,1), dilation_rate=(1,1), 
                  freq_kernel_size=(3,1), temporal_kernel_size=(1,3), dropout_rate=0.1)(x)
    x = TF_ResMax_seq(out_channels = False, strides=(1,1), dilation_rate=(1,1), 
                  freq_kernel_size=(3,1), temporal_kernel_size=(1,3), dropout_rate=0.1)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    x = TF_ResMax_seq(out_channels = 64, strides=(1,1), dilation_rate=(1,1), 
                  freq_kernel_size=(3,1), temporal_kernel_size=(1,3), dropout_rate=0.1)(x)
    x = TF_ResMax_seq(out_channels = False, strides=(1,1), dilation_rate=(1,1), 
                  freq_kernel_size=(3,1), temporal_kernel_size=(1,3), dropout_rate=0.1)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(units=2, activation = 'softmax')(x)

    model = Model(inputs=input_tensor, outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['AUC'])
    return model

def get_BCResMax(input_shape):
    input_tensor = Input(input_shape)
    x1 = Conv2D(filters = 16, kernel_size = (3,3), activation = None)(input_tensor)
    x2 = Conv2D(filters = 16, kernel_size = (3,3), activation = None)(input_tensor)
    x = maximum([x1, x2])
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = BC_ResMax(out_channels = False, strides=(1,1), dilation_rate=(1,1), 
              freq_kernel_size=(3,1), temporal_kernel_size=(1,3), dropout_rate=0.1)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = BC_ResMax(out_channels = 24, strides=(1,1), dilation_rate=(1,1), 
              freq_kernel_size=(3,1), temporal_kernel_size=(1,3), dropout_rate=0.1)(x)
    x = BC_ResMax(out_channels = False, strides=(1,1), dilation_rate=(1,1), 
              freq_kernel_size=(3,1), temporal_kernel_size=(1,3), dropout_rate=0.1)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = BC_ResMax(out_channels = 32, strides=(1,1), dilation_rate=(1,1), 
              freq_kernel_size=(3,1), temporal_kernel_size=(1,3), dropout_rate=0.1)(x)
    x = BC_ResMax(out_channels = False, strides=(1,1), dilation_rate=(1,1), 
              freq_kernel_size=(3,1), temporal_kernel_size=(1,3), dropout_rate=0.1)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = BC_ResMax(out_channels = 48, strides=(1,1), dilation_rate=(1,1), 
              freq_kernel_size=(3,1), temporal_kernel_size=(1,3), dropout_rate=0.1)(x)
    x = BC_ResMax(out_channels = False, strides=(1,1), dilation_rate=(1,1), 
                  freq_kernel_size=(3,1), temporal_kernel_size=(1,3), dropout_rate=0.1)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = BC_ResMax(out_channels = 64, strides=(1,1), dilation_rate=(1,1), 
              freq_kernel_size=(3,1), temporal_kernel_size=(1,3), dropout_rate=0.1)(x)
    x = BC_ResMax(out_channels = False, strides=(1,1), dilation_rate=(1,1), 
              freq_kernel_size=(3,1), temporal_kernel_size=(1,3), dropout_rate=0.1)(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(units=2, activation = 'softmax')(x)
    model = Model(inputs=input_tensor, outputs=x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['AUC'])
    return model



##############
# OFD Model
##############

def Split_img(num_splits, overlap=True):   

    def f(x):

        if num_splits == 1:
            return x

        else:
            freq_num = x.shape[1]
            splits = []

            # if overlap=True
            if overlap:
                if freq_num % (2 * num_splits) == 0:
                    steps = freq_num // (2 * num_splits)
                    length = 2 * steps

                    for i in range(2*num_splits-1):
                        splits.append(x[:,i*steps : i*steps + length,:,:])           
                else:
                    remainder = freq_num % (2 * num_splits)
                    num_pad = (2*num_splits) - remainder
                    x = ZeroPadding2D(padding=( (num_pad-num_pad//2, num_pad//2), (0,0) ) )(x)

                    steps = x.shape[1] // (2 * num_splits)
                    length = steps * 2

                    for i in range(2*num_splits-1):
                        splits.append(x[:, i*steps : i*steps + length, :, :])

            # if overlap=False
            else:
                if freq_num % num_splits == 0:
                    N = freq_num // num_splits
                    for i in range(num_splits):
                        splits.append(x[:, i*N : (i+1)*N, :, :])

                else:
                    remainder = freq_num % num_splits       # remainder is always positive. ('else' case.)
                    num_pad = num_splits - remainder
                    x = ZeroPadding2D(padding=( (num_pad - num_pad//2, num_pad//2), (0,0) ) )(x)

                    length = x.shape[1] // num_splits
                    for i in range(num_splits):
                        splits.append(x[:, i*length : (i+1)*length, :, :])             
            return splits
    return f

def Basic_Block(num_splits:int, overlap=True, filters=32, act='relu', dropout_rate=0.5,
                freq_kernel_size=(3,1), temp_kernel_size=(1,5), dilation_rate=(1,4)):  

    if num_splits > 1:
        def f(x):

            assert act == 'relu' or act == 'mfm'

            if overlap:
                y = Split_img(num_splits = num_splits, overlap=overlap)(x)
                N = 2*num_splits-1

                assert N == len(y)

                if act == 'relu':
                    for j in range(N):
                        y[j] = Conv2D(filters=filters, kernel_size=freq_kernel_size, strides=(1,1),
                                     padding='same', activation=None, kernel_initializer='he_uniform')(y[j])
                        y[j] = BatchNormalization()(y[j])
                        y[j] = relu(y[j])
                        y[j] = Conv2D(filters=filters, kernel_size=freq_kernel_size, strides=(1,1),
                                     padding='same', activation=None, kernel_initializer='he_uniform')(y[j])
                        y[j] = BatchNormalization()(y[j])

                elif act == 'mfm':
                    for j in range(N):
                        temp1 = Conv2D(filters=filters, kernel_size=freq_kernel_size, strides=(1,1),
                                     padding='same', activation=None, kernel_initializer='he_uniform')(y[j])
                        temp2 = Conv2D(filters=filters, kernel_size=freq_kernel_size, strides=(1,1),
                                     padding='same', activation=None, kernel_initializer='he_uniform')(y[j])
                        y[j] = maximum([temp1, temp2])
                        y[j] = BatchNormalization()(y[j])
                        y[j] = Conv2D(filters=filters, kernel_size=freq_kernel_size, strides=(1,1),
                                     padding='same', activation=None, kernel_initializer='he_uniform')(y[j])
                        y[j] = BatchNormalization()(y[j])


                # frequency length for each sub-band 
                length = y[1].shape[1] 
                step = int(length / 2)
                z = []
                for j in range(N):
                    z.append(y[j][:, :step, :, :])
                    z.append(y[j][:, step:, :, :])

                before_concat = []
                before_concat.append(z[0])

                mfm_list = z[1:-1]

                for i in range(len(mfm_list)//2):
                    temp = tf.keras.layers.maximum([mfm_list[2*i], mfm_list[2*i+1]])
                    before_concat.append(temp)

                before_concat.append(z[-1])
                
                assert len(before_concat) == int(2*num_splits)
                
                a = tf.concat(before_concat, axis=1) 

            else:
                y = Split_img(num_splits = num_splits, overlap=overlap)(x)
                N = num_splits

                assert N == len(y)

                if act == 'relu':
                    for j in range(N):
                        y[j] = Conv2D(filters=filters, kernel_size=freq_kernel_size, strides=(1,1),
                                     padding='same', activation=None, kernel_initializer='he_uniform')(y[j])
                        y[j] = BatchNormalization()(y[j])
                        y[j] = relu(y[j])

                        y[j] = Conv2D(filters=filters, kernel_size=freq_kernel_size, strides=(1,1),
                                     padding='same', activation=None, kernel_initializer='he_uniform')(y[j])
                        y[j] = BatchNormalization()(y[j])
                        y[j] = relu(y[j])

                elif act == 'mfm':
                    for j in range(N):
                        temp1 = Conv2D(filters=filters, kernel_size=freq_kernel_size, strides=(1,1),
                                     padding='same', activation=None, kernel_initializer='he_uniform')(y[j])
                        temp2 = Conv2D(filters=filters, kernel_size=freq_kernel_size, strides=(1,1),
                                     padding='same', activation=None, kernel_initializer='he_uniform')(y[j])
                        y[j] = maximum([temp1, temp2])
                        y[j] = BatchNormalization()(y[j])

                        temp1 = Conv2D(filters=filters, kernel_size=freq_kernel_size, strides=(1,1),
                                     padding='same', activation=None, kernel_initializer='he_uniform')(y[j])
                        temp2 = Conv2D(filters=filters, kernel_size=freq_kernel_size, strides=(1,1),
                                     padding='same', activation=None, kernel_initializer='he_uniform')(y[j])
                        y[j] = maximum([temp1, temp2])
                        y[j] = BatchNormalization()(y[j])

                a = tf.concat(y, axis=1)

            x = AveragePooling2D(pool_size=(x.shape[1],1))(x)
            x = DepthwiseConv2D(kernel_size=temp_kernel_size, strides=(1,1), dilation_rate=dilation_rate,
                                padding='same', activation=None, kernel_initializer='he_uniform')(x)
            x = BatchNormalization()(x)
            x = swish(x)

            x = Conv2D(filters=filters, kernel_size=(1,1), strides=(1,1),
                      activation='relu', kernel_initializer='he_uniform')(x)
            x = SpatialDropout2D(dropout_rate)(x)     

            x = UpSampling2D(size=(a.shape[1],1), interpolation='nearest')(x)

            return add([a,x])

        return f

    if num_splits == 0:
        
        def g(x):

            if act == 'relu':
                y = Conv2D(filters=filters, kernel_size=freq_kernel_size, strides=(1,1),
                             padding='same', activation=None, kernel_initializer='he_uniform')(x)
                y = BatchNormalization()(y)
                y = relu(y)
                
                y = Conv2D(filters=filters, kernel_size=freq_kernel_size, strides=(1,1),
                             padding='same', activation=None, kernel_initializer='he_uniform')(y)

                y = BatchNormalization()(y)
                y = relu(y)

            elif act == 'mfm':
                temp1 = Conv2D(filters=filters, kernel_size=freq_kernel_size, strides=(1,1),
                             padding='same', activation=None, kernel_initializer='he_uniform')(x)
                temp2 = Conv2D(filters=filters, kernel_size=freq_kernel_size, strides=(1,1),
                             padding='same', activation=None, kernel_initializer='he_uniform')(x)
                y = maximum([temp1, temp2])
                y = BatchNormalization()(y)
                
                temp1 = Conv2D(filters=filters, kernel_size=freq_kernel_size, strides=(1,1),
                             padding='same', activation=None, kernel_initializer='he_uniform')(y) 
                temp2 = Conv2D(filters=filters, kernel_size=freq_kernel_size, strides=(1,1),
                             padding='same', activation=None, kernel_initializer='he_uniform')(y) 
                y = maximum([temp1, temp2])
                y = BatchNormalization()(y)


            x = AveragePooling2D(pool_size=(x.shape[1],1))(x)
            x = DepthwiseConv2D(kernel_size=temp_kernel_size, strides=(1,1), dilation_rate=dilation_rate,
                                padding='same', activation=None, kernel_initializer='he_uniform')(x)
            x = BatchNormalization()(x)
            x = swish(x)

            x = Conv2D(filters=filters, kernel_size=(1,1), strides=(1,1),
                      activation='relu', kernel_initializer='he_uniform')(x)
            x = SpatialDropout2D(dropout_rate)(x)     

            x = UpSampling2D(size=(y.shape[1],1), interpolation='nearest')(x)


            return add([y,x])

        return g


def get_ofd_model(input_shape, splits:tuple, act:str, overlap:bool, filters=(8,16,32,32,64,128), dropout_rate=0.1):
    
    input_tensor = Input(input_shape)
    
    x = Conv2D(filters=16, kernel_size=(5,5), strides=(1,1), padding='same',
               activation='relu', kernel_initializer='he_uniform')(input_tensor)

    # 1st block
    x = Basic_Block(num_splits=splits[0], filters=filters[0], 
                    overlap=overlap, act=act, dropout_rate=dropout_rate)(x)
    x = MaxPool2D(pool_size=(2,2), strides=(2,2))(x)

    # 2nd block
    x = Basic_Block(num_splits=splits[1], filters=filters[1], 
                    overlap=overlap, act=act, dropout_rate=dropout_rate)(x)
    x = MaxPool2D(pool_size=(2,2), strides=(2,2))(x)

    # 3rd block
    x = Basic_Block(num_splits=splits[2], filters=filters[2], 
                    overlap=overlap, act=act, dropout_rate=dropout_rate)(x)
    x = MaxPool2D(pool_size=(2,2), strides=(2,2))(x)

    # 4th block
    x = Basic_Block(num_splits=splits[3], filters=filters[3], temp_kernel_size=(1,3), 
                    overlap=overlap, act=act, dropout_rate=dropout_rate)(x)
    x = MaxPool2D(pool_size=(2,2), strides=(2,2))(x)

    # 5th block
    x = Basic_Block(num_splits=splits[4], filters=filters[4], freq_kernel_size=(1,1), temp_kernel_size=(1,1),
                    overlap=overlap, act=act, dropout_rate=dropout_rate)(x)
    x = MaxPool2D(pool_size=(2,2), strides=(2,2))(x)

    # 6th block
    x = Basic_Block(num_splits=splits[5], filters=filters[5], freq_kernel_size=(1,1), temp_kernel_size=(1,1),
                    overlap=overlap, act=act, dropout_rate=dropout_rate)(x)
    x = MaxPool2D(pool_size=(2,2), strides=(2,2))(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(units=2, activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=x)
    return model