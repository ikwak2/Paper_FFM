import librosa
import datetime
import tensorflow as tf
import numpy as np
import os
import pickle

from utils.DataLoader import data_loader
from utils.Generator0 import DataGenerator, feature_extract_cqt, evalEER,  evalScore, evalEER_f, evalEER_f2, gen_fname
from models.models import get_ResMax, get_LCNN, get_BCResMax, get_DDWSseq, get_ofd_model, sigmoidal_decay 
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



# use gpu to train
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[5], 'GPU')
    except RuntimeError as e:
        print(e)


# change with your paths if different.        
add2022 = '/Data/data/ADD2022/'
asv2019 = '/Data/data/ASV2019/'

pathset = { 'add2022' : add2022 , 'asv2019':asv2019}
dl = data_loader(pathset)

# pick data (1: ADD, 2: LA)
datapick = '1' 

dl.get_data(data_pick = datapick, tde_pick = 't', to = 't')
dl.get_data(data_pick = datapick, tde_pick = 'd', to = 'd')
dl.get_data(data_pick = datapick, tde_pick = 'e', to = 'e')



###############################################
mname = "BCResMax_ADD_"
#################################################
### FFM Aug ID 7 (Mixup, LF, HF, RF)         
for i in range(3) :
    # feature:
    sr = 16000                                      # sampling rate
    sec = 9.0                                       # adjusted seconds of the sample. 
    feature = "cqt"                                 # feature to be extracted
    filter_scale = 1                                # filter scale factor. Small values (<1) use shorter windows for improved time resolution.
    n_bins = 100                                    # number of frequency bins, starting at fmin 
    fmin = 5                                        # minimum frequency (Hz)

    # training:
    batch_size = 16                                 # batch size
    epoch = 70                                      # number of epochs to run
    dropout_rate = 0.5                              # dropout rate
    human_weight = 5.0                              # assigning weights to classes
    tmp_string = "tmp"                              # used to store intermediate results.

    # data augmentations:
    beta_param = 0.8                                # mixup application rate
    ru = 0.4                                        
    uv = [ru, 1-ru]                                 # determines the sample ratio to apply masking based on the uv value.
    lowpass = [uv, [7,8,9,10,11,12]]                # low frequency masking hyperparameters 
    highpass = [uv, [80,81,82,83,84,85,86,87] ]     # high frequency masking hyperparameters
    ranfilter2 = [uv, 2, [8,9,10,11,12]]            # random frequency masking hyperparameters

    
    params = {'sr': sr,
            'batch_size': batch_size,
            'feature': feature,
            'n_classes': 2,
            'sec': sec,
            'filter_scale': filter_scale,
            'fmin' : fmin,
            'n_bins': int(n_bins),
            'tofile': tmp_string,
            'shuffle': True,
            'beta_param': beta_param,
            'data_dir': add2022,
            'lowpass': lowpass,
            'highpass': highpass,
            'ranfilter2' : ranfilter2 

    }
    params_no_shuffle = {'sr': sr,
                        'batch_size': batch_size,
                        'feature': feature,
                        'n_classes': 2,
                        'sec': sec,
                        'filter_scale': filter_scale,
                        'fmin' : fmin,
                        'n_bins': int(n_bins),
                        'tofile': tmp_string,
                        'shuffle': False,
                        'data_dir': add2022

    }
    training_generator = DataGenerator(dl.train, dl.labels, **params)
    validation_generator = DataGenerator(dl.dev, dl.labels, **params_no_shuffle)
    eval_generator = DataGenerator(dl.eval, dl.labels, **params_no_shuffle)
    input_shape = training_generator.get_input_shape()
    model = get_BCResMax(input_shape)
#    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['AUC'])

    EPOCHS = epoch
    lr = LearningRateScheduler(lambda e: sigmoidal_decay(e, end=EPOCHS))
    class_weight = {0: human_weight, 1: 1.}             ## human: 0, 1: speaker
    history = model.fit_generator(generator=training_generator, #validation_data=track1_generator,
                              epochs=EPOCHS, class_weight=class_weight, callbacks=[lr], verbose=1)
    eer_val = evalEER(validation_generator, model)
    eer_eval = evalEER(eval_generator, model)
    endtxt1 = str(eer_val)[:6] + '.hdf5'
    savefnm = gen_fname(model_name = mname, params = params , dropout_rate = str(dropout_rate), human_weight = str(human_weight), endtxt = endtxt1 )
    savefnm = 'saved_models/' + savefnm
    model.save(savefnm)
    endtxt1 = str(eer_val)[:6] + '.npy'
    fnm = gen_fname(model_name = mname, params = params , dropout_rate = str(dropout_rate), human_weight = str(human_weight), endtxt = endtxt1)
    fnm = 'saved_results/' + fnm
    sc1 = model.predict(eval_generator)
    np.save(fnm, sc1)
    params_save = params
    params_save['model'] = mname
    params_save['human_weight'] = str(human_weight)
    params_save['dropout_rate'] = str(dropout_rate)
    params_save['eer_val'] = str(eer_val)[:6]
    params_save['eer_eval'] = str(eer_eval)[:6]
    params_save['saved_model'] = savefnm
    tnow = datetime.datetime.now()
    params_save['tnow'] = str(tnow)
    print(params_save)
    fnm = 'res_ffm/rec'+str(tnow) + '.pickle'
    with open(fnm, 'wb') as f:
        pickle.dump(params_save, f, pickle.HIGHEST_PROTOCOL)




