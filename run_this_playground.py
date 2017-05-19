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
        
    def get_train_features(self):
        self.features = [self.featurize(a) for a in self.train_audio_paths]
        
    def get_batch(self, index, size, audio_paths, texts):
        # pull necessary info 
        max_length = max([self.features[index+i].shape[0] for i in range(0, size)])
        max_string_length = max([len(self.train_texts[index+i]) for i in range(0, size)])
        
        # initialize the arrays
        X_data = np.zeros([size, max_length, self.feat_dim])
        labels = np.ones([size, max_string_length]) * 28
        input_length = np.zeros([size, 1])
        label_length = np.zeros([size, 1])
        
        # populate the arrays
        for i in range(0, size):
            # X_data, input_length
            feat = self.features[index+i]  
            input_length[i] = feat.shape[0]
            feat = self.normalize(feat) 
            X_data[i, :feat.shape[0], :] = feat

            # y, label_length
            label = np.array(text_to_int_sequence(texts[index + i])) - 1
            labels[i, :len(label)] = label
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
            if self.cur_train_index >= len(self.train_texts)-self.minibatch_size:
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
        # Center using means and std
        return (feature - self.feats_mean) / (self.feats_std + eps)

audio_gen = AudioGenerator(minibatch_size=20)
audio_gen.load_train_data('train_corpus.json')
audio_gen.fit_train(100)
audio_gen.get_train_features()

size = 1000
index = 0

max_length = max([audio_gen.features[index+i].shape[0] for i in range(0, size)])
max_string_length = max([len(audio_gen.train_texts[index+i]) for i in range(0, size)])
        
# initialize the arrays
X_data = np.zeros([size, max_length, audio_gen.feat_dim])
labels = np.ones([size, max_string_length]) * 28
input_length = np.zeros([size, 1])
label_length = np.zeros([size, 1])

for i in range(0, size):
    # X_data, input_length
    feat = audio_gen.features[index+i]  
    feat = audio_gen.normalize(feat) 
    input_length[i] = conv_output_length(max_length, filter_size=11, border_mode='valid', stride=2)
    X_data[i, :feat.shape[0], :] = feat

    # y, label_length
    label = np.array(text_to_int_sequence(audio_gen.train_texts[index + i])) - 1
    labels[i, :len(label)] = label
    label_length[i] = 133

def decode_batch(test_func, audio):
    out = test_func([audio])[0]
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, :], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        # 26 is space, 27 is CTC blank char
        outstr = ''
        for c in out_best:
            if c >= 0 and c < 26:
                outstr += chr(c + ord('a'))
            elif c >= 26:
                outstr += ' '
        ret.append(outstr)
    return ret

class VizCallback(keras.callbacks.Callback):
        
    def on_epoch_end(self, epoch, logs=None):
        print('\n true: ', audio_gen.train_texts[0])
        pred_ints = (K.eval(K.ctc_decode(to_softmax.predict(np.expand_dims(X_data[0], axis=0)), input_length[0])[0][0]) +1).flatten().tolist()
        print('predicted: ',''.join(int_to_text_sequence(pred_ints)))
        print('\n true: ', audio_gen.train_texts[size-1])
        pred_ints = (K.eval(K.ctc_decode(to_softmax.predict(np.expand_dims(X_data[size-1], axis=0)), input_length[0])[0][0]) +1).flatten().tolist()
        print('predicted: ',''.join(int_to_text_sequence(pred_ints)))

char_map_str = """
' 1
<SPACE> 2
a 3
b 4
c 5
d 6
e 7
f 8
g 9
h 10
i 11
j 12
k 13
l 14
m 15
n 16
o 17
p 18
q 19
r 20
s 21
t 22
u 23
v 24
w 25
x 26
y 27
z 28
"""
char_map = {}
index_map = {}
for line in char_map_str.strip().split('\n'):
    ch, index = line.split()
    char_map[ch] = int(index)
    index_map[int(index)] = ch
index_map[2] = ' '

def int_to_text_sequence(int_seq):
    """ Use a character map and convert integer to an text sequence """
    text_seq = []
    for c in int_seq:
        ch = index_map[c]
        text_seq.append(ch)
    return text_seq

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

input_dim=161
output_dim=29
recur_layers=2
filters=1024
kernel_size=11
conv_border_mode='valid'
conv_stride=2
initialization='glorot_uniform' 
minibatch_size=16
   
# define the model
input_data = Input(name='the_input', shape=(None, input_dim))
conv_1d = Conv1D(filters, kernel_size, name='conv1d',
                 padding=conv_border_mode,
                 strides=conv_stride, 
                 kernel_initializer=initialization,
                 activation='relu')(input_data)
output = BatchNormalization(name='bn_conv_1d')(conv_1d)
for r in range(recur_layers):
    output = GRU(filters, activation='relu',
                 name='rnn_{}'.format(r + 1), kernel_initializer=initialization,
                 return_sequences=True)(output)
    bn_layer = BatchNormalization(name='bn_rnn_{}'.format(r + 1))
    output = bn_layer(output)

# transform NN output to character activations
network_output = TimeDistributed(Dense(
    output_dim, name='dense', kernel_initializer=initialization))(output)
y_pred = Activation('softmax', name='softmax')(network_output)

to_softmax = Model(inputs=input_data, outputs=y_pred)

the_labels = Input(name='the_labels', shape=(None,), dtype='float32')
input_lengths = Input(name='input_length', shape=(1,), dtype='int64')
label_lengths = Input(name='label_length', shape=(1,), dtype='int64')

# CTC loss is implemented in a lambda layer
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
    [y_pred, the_labels, input_lengths, label_lengths])

model = Model(inputs=[input_data, the_labels, input_lengths, label_lengths], outputs=loss_out)

sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
# the loss calc occurs elsewhere, so use a dummy lambda func for the loss
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)

viz_cb = VizCallback()

model.fit([X_data, labels, input_length, label_length], np.zeros([size]),
          batch_size=20, epochs=200, callbacks=[viz_cb], validation_split=0.2,
          verbose=1, shuffle=False)