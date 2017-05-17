import numpy as np
import json
import random

from utils import calc_feat_dim, spectrogram_from_file, text_to_int_sequence
from utils import conv_output_length

from keras import backend as K
from keras.optimizers import SGD
from keras.models import Model
import keras.callbacks
from keras.layers import (BatchNormalization, Conv1D, Dense,
                          Input, GRU, TimeDistributed, Activation, Lambda)

RNG_SEED = 123

class AudioGenerator(keras.callbacks.Callback):
    
    def __init__(self, step=10, window=20, max_freq=8000, minibatch_size=30, desc_file=None):
        """
        Params:
            step (int): Step size in milliseconds between windows
            window (int): FFT window size in milliseconds
            max_freq (int): Only FFT bins corresponding to frequencies between
                [0, max_freq] are returned
            desc_file (str, optional): Path to a JSON-line file that contains
                labels and paths to the audio files. If this is None, then
                load metadata right away
        """
        self.feat_dim = calc_feat_dim(window, max_freq)
        self.feats_mean = np.zeros((self.feat_dim,))
        self.feats_std = np.ones((self.feat_dim,))
        self.rng = random.Random(RNG_SEED)
        if desc_file is not None:
            self.load_metadata_from_desc_file(desc_file)
        self.step = step
        self.window = window
        self.max_freq = max_freq
        self.cur_train_index = 0
        self.minibatch_size = minibatch_size
        
    def get_batch(self, index, size, audio_paths, texts):
        
        # pull necessary info from data generator
        features = [self.featurize(a) for a in audio_paths] # change later to [index:index+size]
        input_lengths = [f.shape[0] for f in features]
        max_length = max(input_lengths)
        feature_dim = features[0].shape[1]
        max_string_length = max([len(texts[i]) for i in range(len(texts))])
        
        # initialize the arrays
        X_data = np.zeros([size, max_length, feature_dim])
        labels = np.ones([size, max_string_length]) * 28
        input_length = np.zeros([size, 1])
        label_length = np.zeros([size, 1])
        
        # populate the arrays
        for i in range(0, size):
            # X_data
            feat = features[index + i]  
            feat = self.normalize(feat) # Center using means and std
            X_data[i, :feat.shape[0], :] = feat

            # y, input_length, label_length
            label = np.array(text_to_int_sequence(texts[index + i])) - 1
            labels[i, :len(label)] = label
            input_length[i] = features[index + i].shape[0]
            label_length[i] = len(label)
            
        # repare and return the arrays
        input_length = np.array([conv_output_length(i, filter_size=11, border_mode='valid', stride=2) for i in input_length])
        outputs = {'ctc': np.zeros([size])}
        inputs = {'the_input': X_data, # array; dim: mb_size x max_aud_length x features[0].shape[1]
                  'the_labels': labels, # array; dim: mb_size, time_steps, num_categories
                  'input_length': input_length, # array; dim: mb_size x 1
                  'label_length': label_length # array; dim: mb_size x 1
                 }
        return (inputs, outputs)
        
    def next_train(self):
        while 1:
            ret = self.get_batch(self.cur_train_index, self.minibatch_size, 
                                 self.train_audio_paths, self.train_texts)
            self.cur_train_index += self.minibatch_size
            if self.cur_train_index > 1000:
                self.cur_train_index = 0 
            yield ret
            
    def load_train_data(self, desc_file):
        self.load_metadata_from_desc_file(desc_file, 'train')
    
    def load_metadata_from_desc_file(self, desc_file, partition='train',
                                     max_duration=10.0,):
        """ Read metadata from the description file
            (possibly takes long, depending on the filesize)
        Params:
            desc_file (str):  Path to a JSON-line file that contains labels and
                paths to the audio files
            partition (str): One of 'train', 'validation' or 'test'
            max_duration (float): In seconds, the maximum duration of
                utterances to train or test on
        """
        audio_paths, durations, texts = [], [], []
        with open(desc_file) as json_line_file:
            for line_num, json_line in enumerate(json_line_file):
                try:
                    spec = json.loads(json_line)
                    if float(spec['duration']) > max_duration:
                        continue
                    audio_paths.append(spec['key'])
                    durations.append(float(spec['duration']))
                    texts.append(spec['text'])
                except Exception as e:
                    # Change to (KeyError, ValueError) or
                    # (KeyError,json.decoder.JSONDecodeError), depending on
                    # json module version
                    print('Error reading line #{}: {}'
                                .format(line_num, json_line))
        if partition == 'train':
            self.train_audio_paths = audio_paths
            self.train_durations = durations
            self.train_texts = texts
        elif partition == 'validation':
            self.val_audio_paths = audio_paths
            self.val_durations = durations
            self.val_texts = texts
        elif partition == 'test':
            self.test_audio_paths = audio_paths
            self.test_durations = durations
            self.test_texts = texts
        else:
            raise Exception("Invalid partition to load metadata. " "Must be train/validation/test")
            
    def fit_train(self, k_samples=100):
        """ Estimate the mean and std of the features from the training set
        Params:
            k_samples (int): Use this number of samples for estimation
        """
        k_samples = min(k_samples, len(self.train_audio_paths))
        samples = self.rng.sample(self.train_audio_paths, k_samples)
        feats = [self.featurize(s) for s in samples]
        feats = np.vstack(feats)
        self.feats_mean = np.mean(feats, axis=0)
        self.feats_std = np.std(feats, axis=0)
        
    def featurize(self, audio_clip):
        """ For a given audio clip, calculate the log of its Fourier Transform
        Params:
            audio_clip(str): Path to the audio clip
        """
        return spectrogram_from_file(
            audio_clip, step=self.step, window=self.window,
            max_freq=self.max_freq)

    def normalize(self, feature, eps=1e-14):
        return (feature - self.feats_mean) / (self.feats_std + eps)

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def train(input_dim=161, output_dim=29, recur_layers=3, filters=1024, 
          kernel_size=11, conv_border_mode='valid', conv_stride=2, 
          initialization='glorot_uniform', minibatch_size=30):
    
    # call to data generator
    audio_gen = AudioGenerator(minibatch_size=minibatch_size)
    audio_gen.load_train_data('train_corpus.json')
    audio_gen.fit_train(100)
   
    # define the model
    input_data = Input(name='the_input', shape=(None, input_dim))
    conv_1d = Conv1D(filters, kernel_size, name='conv1d',
                     padding=conv_border_mode,
                     strides=conv_stride, 
                     kernel_initializer=initialization,
                     activation='relu')(input_data)
    output = BatchNormalization(name='bn_conv_1d')(conv_1d)
    for r in range(recur_layers):
        output = GRU(filters, activation='linear',
                     name='rnn_{}'.format(r + 1), kernel_initializer=initialization,
                     return_sequences=True)(output)
        bn_layer = BatchNormalization(name='bn_rnn_{}'.format(r + 1))
        output = bn_layer(output)

    # transform NN output to character activations
    network_output = TimeDistributed(Dense(
        output_dim, name='dense', kernel_initializer=initialization))(output)
    y_pred = Activation('softmax', name='softmax')(network_output)
    Model(inputs=input_data, outputs=y_pred).summary()
    
    labels = Input(name='the_labels', shape=[199], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    
    # CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [y_pred, labels, input_length, label_length])
    
    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_out)
    
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
        
    model.fit_generator(generator=audio_gen.next_train(),
                        steps_per_epoch=100, #2700//batch_size,
                        callbacks=[audio_gen],
                        epochs=1, verbose=1)

if __name__ == '__main__':
	train()