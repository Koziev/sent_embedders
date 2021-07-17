# -*- coding: utf-8 -*-
"""
Использование модели BERT, натренированной кодом train_bert.py, в качестве энкодера
в автоэнкодерной модели.

Для экспериментов по изучению зависимости качества декодирования от сложности BERT.
"""

import random
import numpy as np
import os
import io
import pickle

import sklearn.model_selection
import sentencepiece as spm
from colorclass import Color
import terminaltables

import keras
from keras import Model
from keras_bert import get_base_dict, get_model, gen_batch_inputs
from keras import layers
from keras.layers import RepeatVector
from keras.layers import TimeDistributed


def split_str(s):
    return sp.EncodeAsPieces(s)


def ngrams(s):
    return [s1+s2+s3 for (s1, s2, s3) in zip(s, s[1:], s[2:])]


def jaccard(s1, s2):
    s1 = set(ngrams(s1))
    s2 = set(ngrams(s2))
    return float(len(s1&s2))/float(1e-8+len(s1|s2))


def vectorize(samples, token_dict, max_seq_len):
    nb_samples = len(samples)
    X_tok = np.zeros((nb_samples, max_seq_len), dtype=np.int32)
    X_seg = np.zeros((nb_samples, max_seq_len), dtype=np.int32)
    y = np.zeros((nb_samples, max_seq_len,), dtype=np.int32)

    for isample, sample in enumerate(samples):
        tokens0 = split_str(sample)
        tokens = ['[CLS]'] + tokens0 + ['[SEP]']

        for itoken, token in enumerate(tokens):
            X_tok[isample, itoken] = token_dict[token]
            #X_seg[isample, itok] = 0  # всегда 1 предложение...

        tokens2 = tokens0 + ['[SEP]']
        for itoken, token in enumerate(tokens2):
            y[isample, itoken] = token_dict[token]

    return X_tok, X_seg, y


class VizualizeCallback(keras.callbacks.Callback):
    """
    После каждой эпохи обучения делаем сэмплинг образцов из текущей модели,
    чтобы видеть общее качество декодирования на данный момент.
    """

    def __init__(self, model, test_samples, token_dict, max_seq_len):
        self.model = model
        self.test_samples = test_samples
        self.token_dict = token_dict
        self.index2token = dict((i, t) for t, i in token_dict.items())
        self.max_seq_len = max_seq_len
        self.X_tok, self.X_seg, self.y = vectorize(test_samples, token_dict, max_seq_len)
        self.val_history = []
        self.best_jaccard_score = 0.0

    def on_epoch_end(self, batch, logs={}):
        y_pred = self.model.predict(x=(self.X_tok, self.X_seg), verbose=0)
        pred_samples = []
        for true_text, y in zip(self.test_samples, y_pred):
            y = np.argmax(y, axis=-1)
            pred_tokens = [self.index2token[i] for i in y]
            if '[SEP]' in pred_tokens:
                pred_tokens = pred_tokens[:pred_tokens.index('[SEP]')]

            pred_text = ''.join(pred_tokens).replace('▁', ' ').strip()
            pred_samples.append((true_text, pred_text))

        r_samples = random.sample(pred_samples, k=10)

        table = ['true_output predicted_output'.split()]
        for true_sample, pred_sample in r_samples:
            if true_sample == pred_sample:
                # выдача сетки полностью верная
                output2 = Color('{autogreen}' + pred_sample + '{/autogreen}')
            elif jaccard(true_sample, pred_sample) > 0.5:
                # выдача сетки частично совпала с требуемой строкой
                output2 = Color('{autoyellow}' + pred_sample + '{/autoyellow}')
            else:
                # неправильная выдача сетки
                output2 = Color('{autored}' + pred_sample + '{/autored}')

            table.append((true_sample, output2))

        table = terminaltables.AsciiTable(table)
        print(table.table)

        success_rate = sum((true_sample == pred_sample) for true_sample, pred_sample in pred_samples) / float(len(self.test_samples))
        mean_jac = np.mean([jaccard(true_sample, pred_sample) for true_sample, pred_sample in pred_samples])
        self.val_history.append((success_rate, mean_jac))

        print('{}% samples are inferred without loss, mean jaccard score={}'.format(success_rate*100.0, mean_jac))

        if mean_jac > self.best_jaccard_score:
            self.best_jaccard_score = mean_jac
            with io.open(os.path.join(tmp_dir, 'bert_autoencoder.output.txt'), 'w', encoding='utf-8') as wrt:
                s = table.table
                for k in '\x1b[91m \x1b[92m \x1b[93m \x1b[39m'.split():
                    s = s.replace(k, '')
                wrt.write(s+'\n')


if __name__ == '__main__':
    model_dir = '../tmp'
    tmp_dir = '../tmp'

    with open(os.path.join(model_dir, 'bert.config'), 'rb') as f:
        model_config = pickle.load(f)

    bert_config = model_config['bert']
    spm_model = model_config['spm_model']
    weights_path = model_config['weights_path']
    token_dict = model_config['vocab']
    nb_tokens = len(token_dict)

    max_seq_len = bert_config['seq_len']

    sp = spm.SentencePieceProcessor()
    rc = sp.Load(os.path.join(model_dir, spm_model+'.model'))
    print('SentencePiece model loaded with status={}'.format(rc))

    inputs, bert_output_layer = get_model(training=False, trainable=False, output_layer_num=1, **bert_config)
    bert = Model(inputs=inputs, outputs=bert_output_layer)
    #bert.summary()
    bert.load_weights(weights_path)

    bert_token_dim = bert_output_layer.shape[2]
    print('bert_token_dim={}'.format(bert_token_dim))

    encoder = layers.LSTM(units=64, return_sequences=False)(bert_output_layer)

    decoder = RepeatVector(max_seq_len)(encoder)
    decoder = layers.LSTM(64, return_sequences=True)(decoder)
    decoder = TimeDistributed(layers.Dense(units=nb_tokens, activation='softmax'), name='output')(decoder)

    model = keras.Model(inputs, decoder, name="autodecoder")
    model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam())
    model.summary()
    keras.utils.plot_model(model, to_file=os.path.join(tmp_dir, 'bert_autoencoder.png'), show_shapes=True)
    #exit(0)

    with io.open(os.path.join(tmp_dir, 'bert_autoencoder.model_summary.txt'), 'w', encoding='utf-8') as wrt:
        model.summary(line_length=120, print_fn=lambda s: wrt.write(s+'\n'))

    all_sents = set()
    with io.open('../tmp/assemble_training_corpus_for_bert.txt', 'r', encoding='utf-8') as rdr:
        for line in rdr:
            s = line.strip()
            if s:
                tokens0 = split_str(s)
                if len(tokens0)+2 < max_seq_len:
                    all_sents.add(s)

    all_sents = list(all_sents)

    # ОГРАНИЧИМ ДЛЯ ОТЛАДКИ
    all_sents = all_sents[:200000]

    train_samples, viz_samples = sklearn.model_selection.train_test_split(all_sents, test_size=1000, random_state=123456)

    nb_samples = len(all_sents)
    print('Vectorization of {} samples...'.format(nb_samples))
    X_tok, X_seg, y = vectorize(train_samples, token_dict, max_seq_len)

    weights_path = '../tmp/bert_autoencoder.weights'

    model_checkpoint = keras.callbacks.ModelCheckpoint(weights_path,
                                                       monitor='val_loss',
                                                       verbose=1,
                                                       save_best_only=True,
                                                       save_weights_only=True,
                                                       mode='auto')

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=5, verbose=1, mode='auto', restore_best_weights=True)

    viz = VizualizeCallback(model, viz_samples, token_dict, max_seq_len)

    print('Start training...')
    hist = model.fit(x=(X_tok, X_seg), y=y,
                     epochs=50, validation_split=0.1,
                     callbacks=[viz, model_checkpoint, early_stopping],
                     batch_size=16,  #100,
                     verbose=2)

    with io.open(os.path.join(tmp_dir, 'bert_autoencoder.learning_curve.tsv'), 'w', encoding='utf-8') as wrt:
        wrt.write('epoch\tacc_rate\tmean_jaccard\tval_loss\n')
        for epoch, ((acc_rate, mean_jacc), val_loss) in enumerate(zip(viz.val_history, hist.history['val_loss']), start=1):
            wrt.write('{}\t{}\t{}\t{}\n'.format(epoch, acc_rate, mean_jacc, val_loss))
