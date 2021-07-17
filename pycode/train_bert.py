# -*- coding: utf-8 -*-
"""
Тренировка BERT с заданной конфигурацией на небольшом корпусе
"""

import numpy as np
import keras
import os
import collections
import pickle
import json
import io
import sentencepiece as spm

from keras import Model
import keras_bert
from keras_bert import get_base_dict, get_model, gen_batch_inputs
from keras_bert import compile_model
from keras_bert import Tokenizer

from sklearn.model_selection import train_test_split


batch_size = 200
max_seq_len = 40  # макс. длина предложений, кол-во sentencepiece элементов, т.е. примерно в 3 раза больше, чем слов.
nb_epochs = 100
spm_items = 24000  # при обучении sentencepiece ограничиваем словарь модели таким количеством элементов


tmp_folder = '../tmp'
dataset_path = '../tmp/assemble_training_corpus_for_bert.txt'


def split_str(s):
    #return tokenizer.tokenize(phrase1)
    return sp.EncodeAsPieces(s)
    #return list(itertools.chain(*(word2pieces(word) for word in s.split())))



# --------------- SENTENCEPIECE ----------------------

# Готовим корпус для обучения SentencePiece
sentencepiece_corpus = os.path.join(tmp_folder, 'sentencepiece_corpus.txt')

all_sents = set()
with io.open(sentencepiece_corpus, 'w', encoding='utf-8') as wrt:
    print('Loading samples from {}...'.format(dataset_path))
    with io.open(dataset_path, 'r', encoding='utf-8') as rdr:
        for line in rdr:
            s = line.strip()
            if s and s not in all_sents:
                all_sents.add(s)
                wrt.write('{}\n'.format(s))

spm_name = 'sentencepiece4bert_{}'.format(spm_items)

if not os.path.exists(os.path.join(tmp_folder, spm_name + '.vocab')):
    spm.SentencePieceTrainer.Train(
        '--input={} --model_prefix={} --vocab_size={} --shuffle_input_sentence=true --character_coverage=1.0 --model_type=unigram'.format(
            sentencepiece_corpus, spm_name, spm_items))
    os.rename(spm_name + '.vocab', os.path.join(tmp_folder, spm_name + '.vocab'))
    os.rename(spm_name + '.model', os.path.join(tmp_folder, spm_name + '.model'))

sp = spm.SentencePieceProcessor()
rc = sp.Load(os.path.join(tmp_folder, spm_name + '.model'))
print('SentencePiece model loaded with status={}'.format(rc))

# Загружаем корпус для обучения BERT
print('Loading corpus for BERT...')
sentence_pairs = []
all_words = collections.Counter()

CLS = keras_bert.TOKEN_CLS
SEP = keras_bert.TOKEN_SEP

with io.open('../tmp/assemble_training_corpus_for_bert.txt', 'r', encoding='utf-8') as rdr:
    lines = []
    for line in rdr:
        s = line.strip()
        if s:
            lines.append(s)
        else:
            for phrase1, phrase2 in zip(lines, lines[1:]):
                words1 = split_str(phrase1)
                words2 = split_str(phrase2)
                totlen = len(words1) + len(words2) + 3 # первый токен - [CLS], и еще два [SEP]
                if totlen <= max_seq_len:
                    sentence_pairs.append((words1, words2))
                    all_words.update(words1 + words2)
            lines.clear()

print('vocabulary size={}'.format(len(all_words)))
print('{} samples'.format(len(sentence_pairs)))

# Для визуального контроля сохраним частотный словарь
with io.open(os.path.join(tmp_folder, 'vocab.csv'), 'w', encoding='utf-8') as wrt:
    for word, freq in all_words.most_common():
        wrt.write('{}\t{}\n'.format(word, freq))

# Build token dictionary
token_dict = get_base_dict()  # A dict that contains some special tokens
for pairs in sentence_pairs:
    for token in pairs[0] + pairs[1]:
        if token not in token_dict:
            token_dict[token] = len(token_dict)
token_list = list(token_dict.keys())  # Used for selecting a random word

weights_path = os.path.join(tmp_folder, 'bert.weights')

# Параметры BERT модели сохраним в файле, чтобы потом воссоздать архитектуру
bert_config = {'token_num': len(token_dict),
               'head_num': 4,
               'transformer_num': 1,
               'embed_dim': 32,
               'feed_forward_dim': 256,  # было 100
               'seq_len': max_seq_len,
               'pos_num': max_seq_len,
               'dropout_rate': 0.05
               }

model_config = {'spm_model': spm_name,
                'vocab': token_dict,
                'weights_path': weights_path,
                'bert': bert_config
                }

with open(os.path.join(tmp_folder, 'bert.config'), 'wb') as f:
    pickle.dump(model_config, f)

# Build & train the model
model = get_model(**bert_config)
compile_model(model)
model.summary()

#for layer in model.layers:
#    print('{}: {} --> {}'.format(layer.name, layer.input_shape, layer.output_shape))


def my_generator(samples, batch_size):
    while True:
        start_index = 0
        while (start_index + batch_size) < len(samples):
            yield gen_batch_inputs(samples[start_index: start_index + batch_size],
                                   token_dict,
                                   token_list,
                                   seq_len=max_seq_len,
                                   mask_rate=0.3,
                                   swap_sentence_rate=1.0)
            start_index += batch_size



SEED = 123456
TEST_SHARE = 0.2
samples_train, samples_val = train_test_split(sentence_pairs, test_size=TEST_SHARE, random_state=SEED)

model_checkpoint = keras.callbacks.ModelCheckpoint(weights_path,
                                                   monitor='val_loss',
                                                   verbose=1,
                                                   save_best_only=True,
                                                   save_weights_only=True,
                                                   mode='auto')

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=5, verbose=1, mode='auto', restore_best_weights=True)

print('Start training on {} samples'.format(len(samples_train)))
hist = model.fit(x=my_generator(samples_train, batch_size),
                 steps_per_epoch=len(samples_train) // batch_size,
                 epochs=nb_epochs,
                 validation_data=my_generator(samples_val, batch_size),
                 validation_steps=len(samples_val) // batch_size,
                 callbacks=[model_checkpoint, early_stopping],
                 verbose=2)
#model.load_weights(weights_path)

with open(os.path.join(tmp_folder, 'bert.learning_curve.csv'), 'w') as f:
    for epoch, (l, vl) in enumerate(zip(hist.history['loss'], hist.history['val_loss'])):
        f.write('{}\t{}\t{}\n'.format(epoch+1, l, vl))

# `output_layer` is the last feature extraction layer (the last transformer)
# The input layers and output layer will be returned if `training` is `False`
inputs, output_layer = get_model(training=False, **bert_config)

model2 = Model(inputs=inputs, outputs=output_layer)
model2.summary()

#print('output_layer.output_shape={}'.format(output_layer.output_shape))

print('Copying the weights...')
for layer2 in model2.layers:
    layer2.set_weights(model.get_layer(layer2.name).get_weights())

model2.save_weights(weights_path)
