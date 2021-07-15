"""
Эксперименты с обученной моделью вариационного автоэнкодера для предложений
"""

import os
import io
import pickle

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Bidirectional, TimeDistributed
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import model_from_json

import sentencepiece as spm
import terminaltables
import colorama

from text_vae import Sampling


BOS_TOKEN = '<start>'
EOS_TOKEN = '<end>'


class SentEmbedder:
    def __init__(self):
        pass

    def load(self, model_dir):
        with open(os.path.join(model_dir, 'text_vae.config'), 'rb') as f:
            self.cfg = pickle.load(f)

        self.max_len = self.cfg['max_len']
        self.token2index = self.cfg['token2index']
        self.index2token = dict((i, t) for t, i in self.token2index.items())

        self.bpe_model = spm.SentencePieceProcessor()
        rc = self.bpe_model.Load(os.path.join(model_dir, self.cfg['bpe_model_name'] + '.model'))

        with open(os.path.join(model_dir, 'text_vae.encoder.arch'), 'r') as f:
            self.encoder = model_from_json(f.read(), custom_objects={'Sampling': Sampling})

        self.encoder.load_weights(os.path.join(model_dir, 'text_vae.encoder.weights'))

        with open(os.path.join(model_dir, 'text_vae.decoder.arch'), 'r') as f:
            self.decoder = model_from_json(f.read())

        self.decoder.load_weights(os.path.join(model_dir, 'text_vae.decoder.weights'))

    def encode_sent(self, text):
        data = np.zeros((1, self.max_len), dtype=np.int)
        tokens = [BOS_TOKEN] + self.bpe_model.EncodeAsPieces(text) + [EOS_TOKEN]
        for itoken, token in enumerate(tokens):
            data[0, itoken] = self.token2index[token]

        z_mean, _, _ = self.encoder(data)
        sent_vs = z_mean.numpy()
        return sent_vs[0, :]

    def decode_vector(self, v):
        z_sample = np.expand_dims(v, 0)
        x_decoded = self.decoder.predict(z_sample)
        x_decoded = np.argmax(x_decoded, axis=-1)
        tokens = [self.index2token[i] for i in x_decoded[0]]
        s = ''.join(tokens).replace('▁', ' ').replace(BOS_TOKEN, '').replace(EOS_TOKEN, '').strip()
        return s


def voper(emb, sent1, sent2, sent3):
    v = emb.encode_sent(sent1) + (emb.encode_sent(sent2) - emb.encode_sent(sent3))
    print('"{}" + ("{}" - "{}") => "{}"'.format(sent1, sent2, sent3, emb.decode_vector(v)))


if __name__ == '__main__':
    emb = SentEmbedder()
    emb.load('../tmp/vae')

    print('\nInterpolation')
    samples = ['кошка ловит мышку', 'мышка ест сыр']

    v1 = emb.encode_sent(samples[0])
    v2 = emb.encode_sent(samples[1])

    last_s = ''
    for k in np.linspace(0.0, 1.0, 20):
        v = (1.0-k)*v1 + k*v2
        s = emb.decode_vector(v)
        if s != last_s:
            last_s = s
            print(s)


    print('\nVector arithmetics')
    voper(emb, 'кошка ловит мышку', 'собака не спит', 'собака спит')
    voper(emb, 'кот обожает рыбу', 'мышь ненавидит арбуз', 'мышь обожает арбуз')
    voper(emb, 'я ем', 'ты ел', 'я ел')

