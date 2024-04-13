import tensorflow.keras
from tensorflow.keras import backend as K
from scipy.io import wavfile
import numpy as np
import librosa
import soundfile as sf
from tensorflow.keras.utils import to_categorical
import msgpack
import msgpack_numpy as m
import os
from tensorflow.keras.layers import Conv2D, maximum, add, SeparableConv2D
from sklearn.metrics import roc_curve


m.patch()



##################################################
#
#  DataGenerator_cqt(): Data Generator 
#
##################################################
#  Input:
#    list_IDs : A list of file names for the generator. each file is a sound file.
#    labels : A list of labels for list of IDs. 
#    data_dir : Data directory (ASVspoof2019)
#    batch_size : batch size
#    sr : sampling rate
#    pre_emphasis : pre_emphasis ratio
#    sec : adjusted seconds of the sample. 
#    filter_scale : Filter scale factor. Small values (<1) use shorter windows for improved time resolution.
#    n_bins : Number of frequency bins, starting at fmin 
#    fmin : Minimum frequency. Defaults to C1 ~= 32.70 Hz
#    is_flac : Are sound files in .flac format? 
#    shuffle: shuffle ids
#    ...
#

class DataGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, data_dir, batch_size=32, 
                 sr = 16000, pre_emphasis = 0, feature = 'stft',
                 sec=3.0, mono = 1,
                 frame_size = 0.04, frame_stride = 0.02, NFFT = 512*2,
                 n_classes=2, shuffle=True, tofile = False,
                 filter_scale = 1, n_bins = 100, fmin = 10, 
                 hop_length = 128, win_length = 512, 
                 cutmix = False, cutout = False, specaug = False, specmix = False,
                 beta_param = False, lowpass = False, highpass = False, ranfilter = False, ranfilter2 = False, dropblock = False, filteraug = False, bell = False
                ):
        'Initialization'
        self.data_dir = data_dir
        self.sec = sec
        self.sr = sr
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.mono = mono
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.pre_emphasis = pre_emphasis
        self.feature = feature

        self.frame_size = frame_size
        self.frame_stride = frame_stride
        self.NFFT = NFFT
        self.filter_scale = filter_scale
        self.n_bins = n_bins
        self.fmin = fmin
        
        self.hop_length = hop_length
        self.win_length = win_length
        self.beta_param = beta_param

        self.cutmix = cutmix
        self.cutout = cutout
        self.specaug = specaug
        self.specmix = specmix
        
        self.lowpass = lowpass
        self.highpass = highpass
        self.ranfilter = ranfilter
        self.ranfilter2 = ranfilter2
        self.dropblock = dropblock
        self.filteraug = filteraug
	self.bell = bell


        self.tofile = tofile
        
        if self.feature == "stft" :
            Sig = feature_extract_stft(list_IDs[0], samp_sec=sec, sr = sr, pre_emphasis = pre_emphasis,
                                   frame_size = frame_size, frame_stride = frame_stride,
                                   NFFT = NFFT, min_normalize = False, p_log = True, add_log = 1,
                                   norm = 0, cmvn = 0)[0]
            print("stft")
        elif self.feature == "cqt" :
            Sig = feature_extract_cqt(list_IDs[0], samp_sec=self.sec, sr = self.sr,
                                      pre_emphasis = self.pre_emphasis,
                                      filter_scale = self.filter_scale, n_bins = self.n_bins, fmin = self.fmin)[0]
            print("cqt")
        elif self.feature == "melspec" :
            Sig = feature_extract_melspec(list_IDs[0], samp_sec=self.sec, sr = self.sr,
                                      pre_emphasis = self.pre_emphasis,
                                      hop_length = self.hop_length, win_length = self.win_length, n_mels = self.n_bins)[0]
            print("melspec")
        elif self.feature == "melspec2" :
            Sig = feature_extract_melspec2(list_IDs[0], samp_sec=self.sec, sr = self.sr,
                                      pre_emphasis = self.pre_emphasis,
                                      hop_length = self.hop_length, win_length = self.win_length, n_mels = self.n_bins)[0]
            print("melspec2")
                
        self.M = Sig.shape[0]
        self.N = Sig.shape[1]
        
        print("Input_shape:({},{},1)".format(self.M, self.N,1))
        
        self.on_epoch_end()
        if self.tofile:
            if feature == "stft" :
                self.name = data_dir + 'tmp/' +str(self.tofile) +'sec_' +str(self.sec) + 'sr_' +str(self.sr) + 'fsz_'+str(self.frame_size) + 'fst_'+str(self.frame_stride)+'nfft_' + str(self.NFFT)+'_'

            elif feature == "cqt" :
                self.name = data_dir + 'tmp/' +str(self.tofile) +'sec_' +str(self.sec) + 'sr_' +str(self.sr) + 'fm_'+str(self.fmin) + 'nb_'+str(self.n_bins)+'fs_' + str(self.filter_scale)+'_'
            
            elif feature == "melspec" :
                self.name = data_dir + 'tmp/' +str(self.tofile) +'sec_' +str(self.sec) + 'sr_' +str(self.sr) + 'hop_'+str(self.hop_length) + 'win_'+str(self.win_length)+'nb_' + str(self.n_bins)+'_'
            elif feature == "melspec2" :
                self.name = data_dir + 'tmp/' +str(self.tofile) +'2sec_' +str(self.sec) + 'sr_' +str(self.sr) + 'hop_'+str(self.hop_length) + 'win_'+str(self.win_length)+'nb_' + str(self.n_bins)+'_'
            

        else :
            self.tofile = False
            self.name = False
 
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def get_input_shape(self):
        return (self.M,self.N,1)
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization

        X = np.empty((self.batch_size, self.M, self.N, self.mono))
        y = np.empty((self.batch_size), dtype=int)

        #################################################
        # Generate data
        #################################################
        for j, ID in enumerate(list_IDs_temp):
#            print("dvc:{}".format(self.dvc[dvcs[j]]))
            if self.tofile :
#                print("j: {}, ID:{}, self.name:{}".format(j,ID,self.name))
#                print(self.dvc[dvcs[j]])
#                print("ID:{}, dvc:{}".format(ID,self.dvc[dvcs[j]]))
                write_fnm = self.name + '_'.join(ID.split('/')[-2:]) + '.msgpack'
#                print("j: {}, ID:{}, write_fnm:{}".format(j,ID, write_fnm))
            
                if os.path.exists(write_fnm) : 
#                    print("write_fnm exist: {}".format(write_fnm) )
                    try :
                        with open(write_fnm, 'rb') as data_file:
                            data_loaded = msgpack.unpack(data_file)
                        signal = msgpack.unpackb(data_loaded, object_hook=m.decode)
                    except :
                        if self.feature == "stft" :
                            signal = feature_extract_stft(ID, samp_sec=self.sec, sr = self.sr, pre_emphasis = self.pre_emphasis,
                                                       frame_size = self.frame_size, frame_stride = self.frame_stride,
                                                       NFFT = self.NFFT, min_normalize = False, p_log = True, add_log = 1,
                                                       norm = 0, cmvn = 0)[0]
                        elif self.feature == "cqt" : 
                            signal = feature_extract_cqt(ID, samp_sec=self.sec, sr = self.sr,
                                              pre_emphasis = self.pre_emphasis,
                                              filter_scale = self.filter_scale, n_bins = self.n_bins, fmin = self.fmin)[0]
                        elif self.feature == "melspec" :
                            signal = feature_extract_melspec(ID, samp_sec=self.sec, sr = self.sr,
                                      pre_emphasis = self.pre_emphasis,
                                      hop_length = self.hop_length, win_length = self.win_length, n_mels = self.n_bins)[0]
                        elif self.feature == "melspec2" :
                            signal = feature_extract_melspec2(ID, samp_sec=self.sec, sr = self.sr,
                                      pre_emphasis = self.pre_emphasis,
                                      hop_length = self.hop_length, win_length = self.win_length, n_mels = self.n_bins)[0]
                else :
                    try :
                        if self.feature == "stft" :
                            signal = feature_extract_stft(ID, samp_sec=self.sec, sr = self.sr, pre_emphasis = self.pre_emphasis,
                                                       frame_size = self.frame_size, frame_stride = self.frame_stride,
                                                       NFFT = self.NFFT, min_normalize = False, p_log = True, add_log = 1,
                                                       norm = 0, cmvn = 0)[0]
                        elif self.feature == "cqt" : 
                            signal = feature_extract_cqt(ID, samp_sec=self.sec, sr = self.sr,
                                              pre_emphasis = self.pre_emphasis,
                                              filter_scale = self.filter_scale, n_bins = self.n_bins, fmin = self.fmin)[0]
                        elif self.feature == "melspec" :
                            signal = feature_extract_melspec(ID, samp_sec=self.sec, sr = self.sr,
                                      pre_emphasis = self.pre_emphasis,
                                      hop_length = self.hop_length, win_length = self.win_length, n_mels = self.n_bins)[0]
                        elif self.feature == "melspec2" :
                            signal = feature_extract_melspec2(ID, samp_sec=self.sec, sr = self.sr,
                                      pre_emphasis = self.pre_emphasis,
                                      hop_length = self.hop_length, win_length = self.win_length, n_mels = self.n_bins)[0]
                            
                    except :
                        print(self.data_dir+ID)
                        raise
                        
                    x_enc = msgpack.packb(signal, default=m.encode)
                    with open(write_fnm, 'wb') as outfile:
                        msgpack.pack(x_enc, outfile)
            else :
                    if self.feature == "stft" :
                        signal = feature_extract_stft(ID, samp_sec=self.sec, sr = self.sr, pre_emphasis = self.pre_emphasis,
                                                   frame_size = self.frame_size, frame_stride = self.frame_stride,
                                                   NFFT = self.NFFT, min_normalize = False, p_log = True, add_log = 1,
                                                   norm = 0, cmvn = 0)[0]
                    elif self.feature == "cqt" : 
                        signal = feature_extract_cqt(ID, samp_sec=self.sec, sr = self.sr,
                                          pre_emphasis = self.pre_emphasis,
                                          filter_scale = self.filter_scale, n_bins = self.n_bins, fmin = self.fmin)[0]
                    elif self.feature == "melspec" :
                        signal = feature_extract_melspec(ID, samp_sec=self.sec, sr = self.sr,
                                  pre_emphasis = self.pre_emphasis,
                                  hop_length = self.hop_length, win_length = self.win_length, n_mels = self.n_bins)[0]
                    elif self.feature == "melspec2" :
                        signal = feature_extract_melspec2(ID, samp_sec=self.sec, sr = self.sr,
                                  pre_emphasis = self.pre_emphasis,
                                  hop_length = self.hop_length, win_length = self.win_length, n_mels = self.n_bins)[0]

#            X[j,] = signal.reshape(self.M, self.N, self.mono)

            if self.sec:
                X[j,] = signal.reshape(self.M, self.N, self.mono)
            else:
                X = signal[np.newaxis, :,:,np.newaxis]
                
            # Store class
            y[j] = self.labels[ID]
#        print(y.shape)

        
        nX = X.copy()
        y = to_categorical(np.array(y), self.n_classes)
        ny = y.copy()
        
        if self.beta_param :
            lamda1 = np.random.beta(self.beta_param, self.beta_param, size = self.batch_size)   ## beta_param default : 0.7;  Paper recommendation is 0.6~0.8
            ny = np.array([ lamda1[i]*y[i,:] + (1-lamda1[i])*y[i+1,:] if i != self.batch_size-1  else  lamda1[i]*y[i,:] + (1-lamda1[i])*y[0,:] for i in range(self.batch_size) ])
            nX = np.array([ lamda1[i]*X[i,:,:,:] + (1-lamda1[i])*X[i+1,:,:,:] if i != self.batch_size-1  else  lamda1[i]*X[i,:,:,:] + (1-lamda1[i])*X[0,:,:,:] for i in range(self.batch_size) ])

        nf = nX.shape[1]
        nt = nX.shape[2]
        
        def get_box(lambda_value):
            cut_rat = np.sqrt(1.0 - lambda_value)

            cut_w = int(nf * cut_rat)  # rw
            cut_h = int(nt * cut_rat)  # rh

            cut_x = int(np.random.uniform(low=0, high=nf))  # rx
            cut_y = int(np.random.uniform(low=0, high=nt))  # ry

            boundaryx1 = np.minimum(np.maximum(cut_x - cut_w // 2, 0), nf) #tf.clip_by_value(cut_x - cut_w // 2, 0, IMG_SIZE_x)
            boundaryy1 = np.minimum(np.maximum(cut_y - cut_h // 2, 0), nt) #tf.clip_by_value(cut_y - cut_h // 2, 0, IMG_SIZE_y)
            bbx2 = np.minimum(np.maximum(cut_x + cut_w // 2, 0), nf) #tf.clip_by_value(cut_x + cut_w // 2, 0, IMG_SIZE_x)
            bby2 = np.minimum(np.maximum(cut_y + cut_h // 2, 0), nt) #tf.clip_by_value(cut_y + cut_h // 2, 0, IMG_SIZE_y)

            target_h = bby2 - boundaryy1
            if target_h == 0:
                target_h += 1

            target_w = bbx2 - boundaryx1
            if target_w == 0:
                target_w += 1

            return boundaryx1, boundaryy1, target_h, target_w    
            
        if self.cutmix : 
            lambda1 = np.random.beta(self.cutmix, self.cutmix, size = self.batch_size)   ## beta_param default : 0.7;  Paper recommendation is 0.6~0.8
            for i in range(int(self.batch_size/2)) :
                boundaryx1, boundaryy1, target_h, target_w = get_box(lambda1[i])

                crop1 = X[2*i, boundaryx1:(boundaryx1+target_h), boundaryy1:(boundaryy1+target_w),: ]
                crop2 = X[2*i+1, boundaryx1:(boundaryx1+target_h), boundaryy1:(boundaryy1+target_w),: ]

                nX[2*i, boundaryx1:(boundaryx1+target_h), boundaryy1:(boundaryy1+target_w),: ] = crop2
                nX[2*i+1, boundaryx1:(boundaryx1+target_h), boundaryy1:(boundaryy1+target_w),: ] = crop1
            
                p = (target_h*target_w) / (nf*nt)
                ny[2*i] = (1-p)*y[2*i,:] + p*y[2*i+1,:]
                ny[2*i+1] = p*y[2*i,:] + (1-p)*y[2*i+1,:]
             
            
        if self.cutout : 
            lambda1 = np.random.beta(self.cutout, self.cutout, size = self.batch_size)   
            for i in range(int(self.batch_size/2)) :
                boundaryx1, boundaryy1, target_h, target_w = get_box(lambda1[i])
                nX[2*i, boundaryx1:(boundaryx1+target_h), boundaryy1:(boundaryy1+target_w),: ] = 0
                nX[2*i+1, boundaryx1:(boundaryx1+target_h), boundaryy1:(boundaryy1+target_w),: ] = 0

            

        if self.filteraug :  # [ [3, 6], 6, "linear", [-6,6]]
            n_freq_bin = nf
            n_band, min_bw, f_type, db_range = self.filteraug                           
            n_freq_band = np.random.randint(low = n_band[0], high = n_band[1])         
   
            if n_freq_band > 1 :
                while n_freq_bin - n_freq_band * min_bw + 1 < 0 :  ## if min_bw is too large
                    min_bw -= 1

                band_bndry_freqs = np.sort( np.random.randint(low = 0, high = n_freq_bin - n_freq_band * min_bw + 1,
				                              size = n_freq_band - 1)) + np.arange(1,n_freq_band)*min_bw
                band_bndry_freqs = np.sort(list(set(band_bndry_freqs) | {0,n_freq_bin}))

                if f_type == "step" :
                    band_factors = np.random.uniform(size = (self.batch_size, n_freq_band))  * (db_range[1] - db_range[0]) + db_range[0]
                    band_factors = 10 ** (band_factors / 20)
    
                    freq_filt = np.zeros((self.batch_size, n_freq_bin, nt, 1))
                    for b in range(self.batch_size) :
                        for i in range(n_freq_band):
                            freq_filt[b, band_bndry_freqs[i]:band_bndry_freqs[i + 1], :, :] = band_factors[b, i]

                elif f_type == "linear" :
                    band_factors = np.random.uniform(size = (self.batch_size, n_freq_band+ 1))  * (db_range[1] - db_range[0]) + db_range[0]

                    freq_filt = np.zeros((self.batch_size, n_freq_bin, nt, 1))
                    for b in range(self.batch_size) :
                        for i in range(n_freq_band):
                            freq_filt[b, band_bndry_freqs[i]:band_bndry_freqs[i + 1], :, :] = np.linspace(band_factors[b, i], band_factors[b, i+1], band_bndry_freqs[i+1] - band_bndry_freqs[i])[:,np.newaxis,np.newaxis]         
                    freq_filt = 10 ** (freq_filt / 20)

                nX = nX * freq_filt
                
      
        if self.specaug :  ## ex: specaug = [ [2, 10] , [2, 15] ]
            f_info, t_info = self.specaug
            n_band_f, f_len = f_info
            n_band_t, t_len = t_info
            for i in range(self.batch_size) :
                for _ in range(n_band_f) :
                    b1 = np.random.choice(f_len)
                    loc1 = np.random.choice(nf - b1, size = 1)[0]
                    nX[i, loc1:(loc1 + b1 - 1), :] = 0
                
                for _ in range(n_band_t) :
                    b1 = np.random.choice(t_len)
                    loc1 = np.random.choice(nt - b1, size = 1)[0]
                    nX[i, :, loc1:(loc1 + b1 - 1)] = 0

        if self.specmix :  ## ex: specaug = [ [2, 10] , [2, 15] ]
            f_info, t_info = self.specmix
            n_band_f, f_len = f_info
            n_band_t, t_len = t_info
            for i in range(int(self.batch_size/2)) :
                masked = np.ones( (nf, nt) )
                for _ in range(n_band_f) :
                    b1 = np.random.choice(f_len)
                    loc1 = np.random.choice(nf - b1, size = 1)[0]
                    crop1 = X[2*i, loc1:(loc1 + b1 - 1), :] 
                    crop2 = X[2*i+1, loc1:(loc1 + b1 - 1), :] 
                    
                    nX[2*i, loc1:(loc1 + b1 - 1), :] = crop2
                    nX[2*i+1, loc1:(loc1 + b1 - 1), :] = crop1
                    
                    masked[loc1:(loc1 + b1 - 1), :] = 1
                
                for _ in range(n_band_t) :
                    b1 = np.random.choice(t_len)
                    loc1 = np.random.choice(nt - b1, size = 1)[0]
                    crop1 = X[2*i,:, loc1:(loc1 + b1 - 1)] 
                    crop2 = X[2*i+1,:, loc1:(loc1 + b1 - 1)] 
                    
                    nX[2*i,:, loc1:(loc1 + b1 - 1)] = crop2
                    nX[2*i+1,:, loc1:(loc1 + b1 - 1)] = crop1
                    
                    masked[:,loc1:(loc1 + b1 - 1)] = 1

                p = np.sum(masked) / (nf*nt)
                ny[2*i] = (1-p)*y[2*i,:] + p*y[2*i+1,:]                
                ny[2*i+1] = p*y[2*i,:] + (1-p)*y[2*i+1,:]
                
                
        if self.lowpass :
            uv, lp = self.lowpass
            dec1 = np.random.choice(2, size = self.batch_size, p = uv)
            for i in range(self.batch_size) :
                if dec1[i] == 1 :
                    loc1 = np.random.choice(lp, size = 1)[0]
                    nX[i,:loc1,:] = 0
                    
        if self.highpass :
            uv, hp = self.highpass
            dec1 = np.random.choice(2, size = self.batch_size, p = uv)
            for i in range(self.batch_size) :
                if dec1[i] == 1 :
                    loc1 = np.random.choice(hp, size = 1)[0]
                    nX[i, loc1:,:] = 0

        if self.ranfilter :   ## ex ranfilter = [10,11,12,13,14,15]
            dec1 = np.random.choice(2, size = self.batch_size)
            for i in range(self.batch_size) :
                if dec1[i] == 1 :
                    b1 = np.random.choice(self.ranfilter, size = 1)[0]
                    loc1 = np.random.choice(nf - b1, size = 1)[0]
                    nX[i, loc1:(loc1 + b1 - 1), :] = 0

        if self.dropblock :  ## ex dropblock = [30, 80]
            b1, b2 = self.dropblock
            dec1 = np.random.choice(2, size = self.batch_size)
            for i in range(self.batch_size) :
                if dec1[i] == 1 :
                    loc1 = np.random.choice(nf- b1, size = 1)[0]
                    loc2 = np.random.choice(nt- b2, size = 1)[0]
                    nX[i, loc1:(loc1 + b1 - 1), loc2:(loc2 + b2 - 1)] = 0

        if self.ranfilter2 :   
            uv, raniter, ranf = self.ranfilter2
            dec1 = np.random.choice(raniter, size = self.batch_size, p = uv)
            for i in range(self.batch_size) :
                if dec1[i] > 0 :
                    for j in range(dec1[i]):
                        b1 = np.random.choice(ranf, size = 1)[0]
                        loc1 = np.random.choice(nf - b1, size = 1)[0]
                        nX[i, loc1:(loc1 + b1 - 1), :] = 0
        
        if self.bell:
            def gen_bell(I, c, po, w) :
                return -po * np.exp(-((I-c)/w)**2/2)
            
            p, side, W, po = self.bell
            
            if side == 'low':
                c = range(15, 25)
            elif side == 'high':
                c = range(75, 85)
            elif side == 'all':
                c = range(nf)
            
            if np.random.random() < p:
                w = np.random.choice(range(W//2, W))
                I = np.array(range(nf))
                c = I[np.random.choice(c, self.batch_size)]
                bellf = gen_bell(I[np.newaxis, :], c[:, np.newaxis], po=po, w=w)
                bellfilter = 10**(bellf / 10)
                nX = nX * bellfilter[:,:,np.newaxis, np.newaxis]	
                    
        return nX, ny

            
            
    

class DataGenerator_eval(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, data_dir, batch_size=32, 
                 sr = 16000, pre_emphasis = 0, feature = 'stft',
                 sec=3.0, mono = 1,
                 frame_size = 0.04, frame_stride = 0.02, NFFT = 512*2,
                 n_classes=2, shuffle=True, tofile = False,
                 filter_scale = 1, n_bins = 100, fmin = 10, 
                 hop_length = 128, win_length = 512, lowpass = False, highpass = False
                ):
        'Initialization'
        self.data_dir = data_dir
        self.sec = sec
        self.sr = sr
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.mono = mono
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.pre_emphasis = pre_emphasis
        self.feature = feature

        self.frame_size = frame_size
        self.frame_stride = frame_stride
        self.NFFT = NFFT
        self.filter_scale = filter_scale
        self.n_bins = n_bins
        self.fmin = fmin
        self.lowpass = lowpass
        self.highpass = highpass
        
        self.hop_length = hop_length
        self.win_length = win_length

#        self.norm = norm
#        self.add_log = add_log
#        self.cmvn = cmvn

        if self.feature == "stft" :
            Sig = feature_extract_stft(list_IDs[0], samp_sec=sec, sr = sr, pre_emphasis = pre_emphasis,
                                   frame_size = frame_size, frame_stride = frame_stride,
                                   NFFT = NFFT, min_normalize = False, p_log = True, add_log = 1,
                                   norm = 0, cmvn = 0)[0]
        else :
            Sig = feature_extract_cqt(list_IDs[0], samp_sec=self.sec, sr = self.sr,
                                      pre_emphasis = self.pre_emphasis,
                                      filter_scale = self.filter_scale, n_bins = self.n_bins, fmin = self.fmin)[0]
        self.M = Sig.shape[0]
        self.N = Sig.shape[1]
                
        self.on_epoch_end()
        if tofile:
            self.tofile = tofile
            if feature == "stft" :
                self.name = data_dir + 'tmp/' +str(self.tofile) +'sec_' +str(self.sec) + 'sr_' +str(self.sr) + 'fsz_'+str(self.frame_size) + 'fst_'+str(self.frame_stride)+'nfft_' + str(self.NFFT)+'_'
#+'.msgpack'
            elif feature == "cqt" :
                self.name = data_dir + 'tmp/' +str(self.tofile) +'sec_' +str(self.sec) + 'sr_' +str(self.sr) + 'fm_'+str(self.fmin) + 'nb_'+str(self.n_bins)+'fs_' + str(self.filter_scale)+'_'
            
            elif feature == "melspec" :
                self.name = data_dir + 'tmp/' +str(self.tofile) +'sec_' +str(self.sec) + 'sr_' +str(self.sr) + 'hop_'+str(self.hop_length) + 'win_'+str(self.win_length)+'nb_' + str(self.n_bins)+'_'
            elif feature == "melspec2" :
                self.name = data_dir + 'tmp/' +str(self.tofile) +'2sec_' +str(self.sec) + 'sr_' +str(self.sr) + 'hop_'+str(self.hop_length) + 'win_'+str(self.win_length)+'nb_' + str(self.n_bins)+'_'
        else :
            self.tofile = False
            self.name = False
 
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X = self.__data_generation(list_IDs_temp)

        return X

    def get_input_shape(self):
        return (self.M,self.N,1)
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization

        X = np.empty((self.batch_size, self.M, self.N, self.mono))

        #################################################
        # Generate data
        #################################################
        for j, ID in enumerate(list_IDs_temp):
#            print("j: {}, ID:{}, dvc:{}, dvcs:{}".format(j,ID, self.dvc, dvcs))
#            print("dvc:{}".format(self.dvc[dvcs[j]]))
            if self.name :
#                print(self.dvc[dvcs[j]])
#                print("ID:{}, dvc:{}".format(ID,self.dvc[dvcs[j]]))
                write_fnm = self.name + '_'.join(ID.split('/')[-2:]) + '.msgpack'
                if os.path.exists(write_fnm) :
                    try :
                        with open(write_fnm, 'rb') as data_file:
                            data_loaded = msgpack.unpack(data_file)
                        signal = msgpack.unpackb(data_loaded, object_hook=m.decode)
                    except :
                        print("reading msgpack error:{}. read from data.".format(write_fnm))
                        if self.feature == "stft" :
                            signal = feature_extract_stft(ID, samp_sec=self.sec, sr = self.sr, pre_emphasis = self.pre_emphasis,
                                                       frame_size = self.frame_size, frame_stride = self.frame_stride,
                                                       NFFT = self.NFFT, min_normalize = False, p_log = True, add_log = 1,
                                                       norm = 0, cmvn = 0)[0]
                        elif self.feature == "cqt" : 
                            signal = feature_extract_cqt(ID, samp_sec=self.sec, sr = self.sr,
                                              pre_emphasis = self.pre_emphasis,
                                              filter_scale = self.filter_scale, n_bins = self.n_bins, fmin = self.fmin)[0]
                        elif self.feature == "melspec" :
                            signal = feature_extract_melspec(ID, samp_sec=self.sec, sr = self.sr,
                                      pre_emphasis = self.pre_emphasis,
                                      hop_length = self.hop_length, win_length = self.win_length, n_mels = self.n_bins)[0]
                        elif self.feature == "melspec2" :
                            signal = feature_extract_melspec2(ID, samp_sec=self.sec, sr = self.sr,
                                      pre_emphasis = self.pre_emphasis,
                                      hop_length = self.hop_length, win_length = self.win_length, n_mels = self.n_bins)[0]
                else :
                    try :
                        if self.feature == "stft" :
                            signal = feature_extract_stft(ID, samp_sec=self.sec, sr = self.sr, pre_emphasis = self.pre_emphasis,
                                                       frame_size = self.frame_size, frame_stride = self.frame_stride,
                                                       NFFT = self.NFFT, min_normalize = False, p_log = True, add_log = 1,
                                                       norm = 0, cmvn = 0)[0]
                        elif self.feature == "cqt" : 
                            signal = feature_extract_cqt(ID, samp_sec=self.sec, sr = self.sr,
                                              pre_emphasis = self.pre_emphasis,
                                              filter_scale = self.filter_scale, n_bins = self.n_bins, fmin = self.fmin)[0]
                        elif self.feature == "melspec" :
                            signal = feature_extract_melspec(ID, samp_sec=self.sec, sr = self.sr,
                                      pre_emphasis = self.pre_emphasis,
                                      hop_length = self.hop_length, win_length = self.win_length, n_mels = self.n_bins)[0]
                        elif self.feature == "melspec2" :
                            signal = feature_extract_melspec2(ID, samp_sec=self.sec, sr = self.sr,
                                      pre_emphasis = self.pre_emphasis,
                                      hop_length = self.hop_length, win_length = self.win_length, n_mels = self.n_bins)[0]
                    except :
                        print(self.data_dir+ID)
                        raise

                    x_enc = msgpack.packb(signal, default=m.encode)
                    with open(write_fnm, 'wb') as outfile:
                        msgpack.pack(x_enc, outfile)
                        
            else :
                    if self.feature == "stft" :
                        signal = feature_extract_stft(ID, samp_sec=self.sec, sr = self.sr, pre_emphasis = self.pre_emphasis,
                                                   frame_size = self.frame_size, frame_stride = self.frame_stride,
                                                   NFFT = self.NFFT, min_normalize = False, p_log = True, add_log = 1,
                                                   norm = 0, cmvn = 0)[0]
                    elif self.feature == "cqt" : 
                        signal = feature_extract_cqt(ID, samp_sec=self.sec, sr = self.sr,
                                          pre_emphasis = self.pre_emphasis,
                                          filter_scale = self.filter_scale, n_bins = self.n_bins, fmin = self.fmin)[0]
                    elif self.feature == "melspec" :
                        signal = feature_extract_melspec(ID, samp_sec=self.sec, sr = self.sr,
                                  pre_emphasis = self.pre_emphasis,
                                  hop_length = self.hop_length, win_length = self.win_length, n_mels = self.n_bins)[0]
                    elif self.feature == "melspec2" :
                        signal = feature_extract_melspec2(ID, samp_sec=self.sec, sr = self.sr,
                                  pre_emphasis = self.pre_emphasis,
                                  hop_length = self.hop_length, win_length = self.win_length, n_mels = self.n_bins)[0]

            X[j,] = signal.reshape(self.M, self.N, self.mono)

#        print(y.shape)

        nX = X.copy()

        if self.lowpass :
            for i in range(self.batch_size) :
                loc1 = np.random.choice(self.lowpass, size = 1)[0]
                nX[i,:loc1,:] = nX[i,:loc1,:] * 0.01
                    
        if self.highpass :
            nf = nX.shape[2]
            for i in range(self.batch_size) :
                loc1 = np.random.choice(self.highpass, size = 1)[0]
                nX[i, loc1:,:] = nX[i, loc1:,:] * 0.01


        return nX

##################################################
#
#  feature_extract_melspec(): extract a mel spectrogram feature 
#
##################################################

def feature_extract_melspec(fnm, samp_sec=3, sr = 16000, pre_emphasis = 0, hop_length=128, win_length = 512, n_mels = 80):
            
    if fnm.split('.')[-1] == 'npy' :
        data = np.load(fnm)
        sample_rate = sr
    else : 
        data, sample_rate = sf.read(fnm, dtype = 'int16')
    data = data * 1.0

    if samp_sec:
        if len(data) > sample_rate * samp_sec : 
            n_samp = len(data) // int(sample_rate * samp_sec)
            signal = []
            for i in range(n_samp) :
                signal.append(data[ int(sample_rate * samp_sec)*i:(int(sample_rate * samp_sec)*(i+1))]) 
        else :
            n_samp = 1
            signal = np.zeros(int(sample_rate*samp_sec,))
            for i in range(int(sample_rate * samp_sec) // len(data)) :
                signal[(i)*len(data):(i+1)*len(data)] = data
            num_last = int(sample_rate * samp_sec) - len(data)*(i+1)
            signal[(i+1)*len(data):int(sample_rate * samp_sec)] = data[:num_last]
            signal = [signal]    
    else:
        n_samp = 1
        signal = [data]   
        
    Sig = []
    for i in range(n_samp) :
        if pre_emphasis :
            emphasized_signal = np.append(signal[i][0], signal[i][1:] - pre_emphasis * signal[i][:-1])
        else :
            emphasized_signal = signal[i] 
            
        Sig.append(librosa.power_to_db(librosa.feature.melspectrogram(y=emphasized_signal, sr= 16000, n_mels=n_mels, n_fft=win_length, hop_length=hop_length, win_length=win_length)))

    return Sig

##################################################
#
#  feature_extract_melspec(): extract a mel spectrogram feature 
#
##################################################

def feature_extract_melspec2(fnm, samp_sec=3, sr = 16000, pre_emphasis = 0, hop_length=128, win_length = 512, n_mels = 80, dtype1 = 'int16'):
            
    if fnm.split('.')[-1] == 'npy' :
        data = np.load(fnm)
        sample_rate = sr
    else : 
        data, sample_rate = sf.read(fnm, dtype = 'int16')
    data = data * 1.0

    if samp_sec:
        if len(data) > sample_rate * samp_sec : 
            n_samp = len(data) // int(sample_rate * samp_sec)
            signal = []
            for i in range(n_samp) :
                signal.append(data[ int(sample_rate * samp_sec)*i:(int(sample_rate * samp_sec)*(i+1))]) 
        else :
            n_samp = 1
            signal = np.zeros(int(sample_rate*samp_sec,))
            for i in range(int(sample_rate * samp_sec) // len(data)) :
                signal[(i)*len(data):(i+1)*len(data)] = data
            num_last = int(sample_rate * samp_sec) - len(data)*(i+1)
            signal[(i+1)*len(data):int(sample_rate * samp_sec)] = data[:num_last]
            signal = [signal]    
    else:
        n_samp = 1
        signal = [data]   
        
    Sig = []
    for i in range(n_samp) :
        if pre_emphasis :
            emphasized_signal = np.append(signal[i][0], signal[i][1:] - pre_emphasis * signal[i][:-1])
        else :
            emphasized_signal = signal[i] 
            
        Sig.append(np.log(librosa.feature.melspectrogram(y=emphasized_signal, sr= 16000, n_mels=n_mels, n_fft=win_length, hop_length=hop_length, win_length=win_length)))

    return Sig

##################################################
#
#  feature_extract_stft(): extract a STFT feature 
#
##################################################

def feature_extract_stft(fnm, samp_sec=3, sr = 16000, pre_emphasis = 0, frame_size = 0.04, 
                    frame_stride = 0.04*.5, NFFT = 512*2, min_normalize = False, p_log = True,
                     norm = 0, add_log = 1, cmvn = 0) :

    if fnm.split('.')[-1] == 'npy' :
        data = np.load(fnm)
        sample_rate = sr
    else : 
        data, sample_rate = sf.read(fnm, dtype = 'int16')
#    data, sample_rate = sf.read(fnm, dtype = 'int16')

    if samp_sec:
        if len(data) > sample_rate * samp_sec : 
            n_samp = len(data) // int(sample_rate * samp_sec)
            signal = []
            for i in range(n_samp) :
                signal.append(data[ int(sample_rate * samp_sec)*i:(int(sample_rate * samp_sec)*(i+1))]) 
        else :
            n_samp = 1
            signal = np.zeros(int(sample_rate*samp_sec,))
            for i in range(int(sample_rate * samp_sec) // len(data)) :
                signal[(i)*len(data):(i+1)*len(data)] = data
            num_last = int(sample_rate * samp_sec) - len(data)*(i+1)
            signal[(i+1)*len(data):int(sample_rate * samp_sec)] = data[:num_last]
            signal = [signal]    
    else:
        n_samp = 1
        signal = [data]
    
    Sig = []
    for i in range(n_samp) :
        if pre_emphasis :
            emphasized_signal = np.append(signal[i][0], signal[i][1:] - pre_emphasis * signal[i][:-1])
        else :
            emphasized_signal = signal[i]
        frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples

        signal_length = len(emphasized_signal)
        frame_length = int(round(frame_length))
        frame_step = int(round(frame_step))

        num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame
        pad_signal_length = num_frames * frame_step + frame_length
        z = np.zeros((pad_signal_length - signal_length))
        #indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
        pad_signal = np.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal
        indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
        frames = pad_signal[indices.astype(np.int32, copy=False)]
        frames *= np.hamming(frame_length)
    
        mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
        pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

#        if min_normalize :
#            pow_frames -= np.min(pow_frames, axis = 1).reshape(1,-1).T
    
        if p_log :
            pow_frames = np.log10(pow_frames + add_log)
            
#        if norm : 
#            pow_frames = librosa.util.normalize(pow_frames, norm = norm)
            
#        if cmvn :
#            pow_frames = speechpy.processing.cmvnw(pow_frames, win_size=1727, variance_normalization=True) 

        Sig.append(pow_frames.T)
    return Sig

##################################################
#
#  feature_extract_cqt(): extract a CQT feature 
#
##################################################
#  Input:
#    fnm : a file name
#    samp_sec : adjusted seconds of the sample. 
#    pre_emphasis : pre_emphasis ratio
#    filter_scale : Filter scale factor. Small values (<1) use shorter windows for improved time resolution.
#    n_bins : Number of frequency bins, starting at fmin 
#    fmin : Minimum frequency. Defaults to C1 ~= 32.70 Hz
#
#  output:
#    Constant-Q value each frequency at each time.

def feature_extract_cqt(fnm, samp_sec=3, sr = 16000, pre_emphasis = 0, filter_scale = 1, n_bins = 100, fmin = 10) :
            
    if fnm.split('.')[-1] == 'npy' :
        data = np.load(fnm)
        sample_rate = sr
    else : 
        data, sample_rate = sf.read(fnm, dtype = 'int16')
#    data, sample_rate = sf.read(fnm, dtype = 'int16')
    data = data * 1.0

    if samp_sec:
        if len(data) > sample_rate * samp_sec : 
            n_samp = len(data) // int(sample_rate * samp_sec)
            signal = []
            for i in range(n_samp) :
                signal.append(data[ int(sample_rate * samp_sec)*i:(int(sample_rate * samp_sec)*(i+1))]) 
        else :
            n_samp = 1
            signal = np.zeros(int(sample_rate*samp_sec,))
            for i in range(int(sample_rate * samp_sec) // len(data)) :
                signal[(i)*len(data):(i+1)*len(data)] = data
            num_last = int(sample_rate * samp_sec) - len(data)*(i+1)
            signal[(i+1)*len(data):int(sample_rate * samp_sec)] = data[:num_last]
            signal = [signal]    
    else:
        n_samp = 1
        signal = [data]   
        
    Sig = []
    for i in range(n_samp) :
        if pre_emphasis :
            emphasized_signal = np.append(signal[i][0], signal[i][1:] - pre_emphasis * signal[i][:-1])
        else :
            emphasized_signal = signal[i] 
            
        Sig.append(np.log(np.abs(librosa.cqt(emphasized_signal, sr=sample_rate,
                                             filter_scale = filter_scale, n_bins=n_bins,
                                             fmin = fmin))+1))
    return Sig

def evalScore(gen, model) :
    batchsize = gen.batch_size
    sc = model.predict_generator(gen)
    return sc[:,1]

def evalEER(gen, model) :
    batchsize = gen.batch_size
    
    label_valid = [ gen.labels[fnm] for fnm in gen.list_IDs ]
    label_valid = label_valid[:batchsize* (len(label_valid) // batchsize) ]

    sc = model.predict_generator(gen)
    fpr, tpr, threshold = roc_curve(label_valid, sc[:,1])
    fnr = 1-tpr
    minloc = np.absolute(fnr-fpr).argmin()
    eer = (fpr[minloc] + fnr[minloc]) / 2
    print('EER: {}'.format(eer))

    return eer

## generate filename

def gen_fname(model_name, params, dropout_rate = 'na', human_weight='na', endtxt = '.txt') :
    str1 = 'model_name!' + model_name + '!dropout_rate!' + str(dropout_rate) + '!human_weight!' + str(human_weight) + '!' 
    for (i,j) in params.items() :
        if (i == "lowpass") or (i == "highpass") or (i == "ranfilter2") or (i == "ranfilter") or (i == "dropblock") or (i =="filteraug"):
            str1 = str1 + str(i) + '!--!'
        elif (i != 'data_dir') and (i != 'n_classes') and (i != 'batch_size') and (i !='sr') and (i != 'tofile') :
            str1 = str1 + str(i) + '!' + str(j) + '!'
    str1 += endtxt
    return str1


def eval_track2(valid_gen, eval_gen, model) :

    batchsize = valid_gen.batch_size

    label_valid = [ valid_gen.labels[fnm] for fnm in valid_gen.list_IDs ]
    label_valid = label_valid[:batchsize* (len(label_valid) // batchsize) ]

    sc = model.predict_generator(valid_gen)
    fpr, tpr, threshold = roc_curve(label_valid, sc[:,1])
    fnr = 1-tpr
    minloc = np.absolute(fnr-fpr).argmin()
    eer = (fpr[minloc] + fnr[minloc]) / 2
#    print('EER: {}'.format(eer))

    threshold1 = threshold[minloc]
    sc1 = evalScore(eval_gen, model)
    
    acc1= 0
    tot_n = len(sc1)
    for i in range(tot_n) :
        acc1 += (sc1[i]>threshold1) 
        
    print("Error Rate:{}, threshold:{}".format(1- (acc1 / tot_n), threshold1))
    return 1 - acc1 / tot_n


def evalEER_f(gen, fnms) :

    st = False
    nf = len(fnms)
    for f in fnms:
        sc1 = np.load(f)
        if st :
            sc += sc1 / nf
        else :
            sc = sc1 / nf
            st = True

    batchsize = gen.batch_size
    
    label_valid = [ gen.labels[fnm] for fnm in gen.list_IDs ]
    label_valid = label_valid[:batchsize* (len(label_valid) // batchsize) ]

#    sc = model.predict_generator(gen)
    fpr, tpr, threshold = roc_curve(label_valid, sc)
    fnr = 1-tpr
    minloc = np.absolute(fnr-fpr).argmin()
    eer = (fpr[minloc] + fnr[minloc]) / 2
    print('EER: {}'.format(eer))

    return eer

def evalEER_f2(gen, fnms) :

    sc = []
    scstd = []
    nf = len(fnms)
    for f in fnms:
        sc1 = np.load(f)
        sc.append(sc1)
        scstd.append(sc1.std())

    scstd = np.array(scstd) / sum(scstd)
    sc2 = np.zeros( (len(sc1)) )
    for (sc1,std1) in zip(sc,scstd) :
        sc2 += sc1 / std1

    batchsize = gen.batch_size
    
    label_valid = [ gen.labels[fnm] for fnm in gen.list_IDs ]
    label_valid = label_valid[:batchsize* (len(label_valid) // batchsize) ]

#    sc = model.predict_generator(gen)
    fpr, tpr, threshold = roc_curve(label_valid, sc2)
    fnr = 1-tpr
    minloc = np.absolute(fnr-fpr).argmin()
    eer = (fpr[minloc] + fnr[minloc]) / 2
    print('EER: {}'.format(eer))

    return eer
