"""
Тренировка модели вариационного автоэнкодера для предложений.
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


BOS_TOKEN = '<start>'
EOS_TOKEN = '<end>'


class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim)) * 0.5
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def train_bpe_model(all_sents, params, model_dir):
    spm_items = params['spm_items']

    # Готовим корпус для обучения SentencePiece
    sentencepiece_corpus = os.path.join(tmp_dir, 'sentencepiece_corpus.txt')

    with io.open(sentencepiece_corpus, 'w', encoding='utf-8') as wrt:
        for s in all_sents:
            wrt.write('{}\n'.format(s))

    spm_name = 'text_vae1_sentencepiece'

    print('Start training bpe model "{}"'.format(spm_name))
    spm.SentencePieceTrainer.Train(
        '--input={} --model_prefix={} --vocab_size={} --shuffle_input_sentence=true --character_coverage=1.0 --model_type=unigram'.format(
            sentencepiece_corpus, spm_name, spm_items))
    os.rename(spm_name + '.vocab', os.path.join(model_dir, spm_name + '.vocab'))
    os.rename(spm_name + '.model', os.path.join(model_dir, spm_name + '.model'))

    print('bpe model "{}" ready'.format(spm_name))
    return spm_name


def load_bpe_model(spm_name):
    sp = spm.SentencePieceProcessor()
    rc = sp.Load(os.path.join(tmp_dir, spm_name + '.model'))
    print('bpe model "{}" loaded with status={}'.format(spm_name, rc))
    return sp


def load_sents():
    sents = set()
    with io.open('/home/inkoziev/polygon/chatbot/tmp/pqa_all.dat', 'r', encoding='utf-8') as rdr:
        for line in rdr:
            s = line.strip()
            if s:
                sents.add(s)

    with io.open('/home/inkoziev/polygon/chatbot/tmp/interpreter_samples.tsv', 'r', encoding='utf-8') as rdr:
        rdr.readline()
        for line in rdr:
            tx = line.split('\t')
            for s in tx[0].split('|'):
                sents.add(s.strip())
            sents.add(tx[1].strip())

    return list(sents)


def load_samples(sents, bpe_model, max_phrase_charlen):
    samples = []
    all_tokens = set()
    max_len = 0

    for s in sents:
        if len(s) <= max_phrase_charlen:
            tokens = [BOS_TOKEN] + bpe_model.EncodeAsPieces(s) + [EOS_TOKEN]
            all_tokens.update(tokens)
            samples.append((s, tokens))
            max_len = max(max_len, len(tokens))

    token2index = dict((t, i) for i, t in enumerate(all_tokens, start=1))
    token2index[''] = 0

    computed_params = dict()
    computed_params['token2index'] = token2index
    computed_params['max_len'] = max_len
    computed_params['start_token_index'] = token2index[BOS_TOKEN]
    computed_params['end_token_index'] = token2index[EOS_TOKEN]
    computed_params['nb_tokens'] = len(token2index)

    vectorized_samples = np.zeros((len(samples), max_len), dtype=np.int)
    for irow, (text, tokens) in enumerate(samples):
        for itoken, token in enumerate(tokens):
            vectorized_samples[irow, itoken] = token2index[token]

    return samples, vectorized_samples, computed_params


def ngrams(s0, n):
    s = '\b' + s0 + '\n'
    return set(u''.join(z) for z in zip(*[s[i:] for i in range(n)]))


def jaccard(s1, s2, shingle_len):
    shingles1 = ngrams(s1.lower(), shingle_len)
    shingles2 = ngrams(s2.lower(), shingle_len)
    return float(len(shingles1&shingles2))/float(1e-8+len(shingles1|shingles2))


def print_red_line(msg):
    print(colorama.Fore.RED + msg + colorama.Fore.RESET)


def print_green_line(msg):
    print(colorama.Fore.GREEN + msg + colorama.Fore.RESET)


class VisualizeCallback(keras.callbacks.Callback):
    def __init__(self, X_val, samples_val, vae, computed_params, save_dir):
        self.epoch = 0
        self.save_dir = save_dir
        self.X = X_val
        self.samples = samples_val
        self.vae = vae
        self.index2token = dict((i, c) for c, i in computed_params['token2index'].items())
        self.best_val_acc = -np.inf  # для сохранения самой точной модели
        self.wait = 0  # для early stopping по критерию общей точности
        self.stopped_epoch = 0
        self.patience = 5

    def decode_text(self, x):
        tx = [self.index2token[i] for i in x if i != 0]
        if tx[0] == BOS_TOKEN:
            tx = tx[1:]
        if tx[-1] == EOS_TOKEN:
            tx = tx[:-1]
        s = ''.join(tx).replace('▁', ' ').strip()
        return s

    def on_epoch_end(self, batch, logs={}):
        self.epoch += 1

        z_mean, _, _ = self.vae.encoder(self.X)
        text_vs = z_mean.numpy()

        x_decoded = self.vae.decoder.predict(text_vs)
        x_decoded = np.argmax(x_decoded, axis=-1)

        viztable = ['true_text reconstructed_text, similarity'.split()]
        jsims = []
        for i in np.random.permutation(np.arange(self.X.shape[0])):
            true_text = self.samples[i][0]
            pred_text = self.decode_text(x_decoded[i])
            sim = jaccard(true_text, pred_text, 3)
            jsims.append(sim)

            if len(viztable) < 10:
                viztable.append([true_text, pred_text, sim])

        print(terminaltables.AsciiTable(viztable).table)

        val_acc = np.mean(jsims)

        if val_acc > self.best_val_acc:
            print_green_line('\nInstance accuracy improved from {} to {}\n'.format(self.best_val_acc, val_acc))
            self.best_val_acc = val_acc
            self.vae.encoder.save_weights(os.path.join(self.save_dir, 'text_vae.encoder.weights'))
            self.vae.decoder.save_weights(os.path.join(self.save_dir, 'text_vae.decoder.weights'))
            self.wait = 0
        else:
            print('\nTotal instance accuracy={} did not improve (current best acc={})\n'.format(val_acc, self.best_val_acc))
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = self.epoch
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

    def get_best_accuracy(self):
        return self.best_val_acc


"""
## Define the VAE as a `Model` with a custom `train_step`
"""
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    #keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                    keras.losses.sparse_categorical_crossentropy(data, reconstruction), axis=(1,)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


def create_model(params, computed_params):
    latent_dim = params['latent_dim']
    token_dim = params['token_dim']
    max_len = computed_params['max_len']
    nb_tokens = computed_params['nb_tokens']

    """
    ## Build the encoder
    """
    encoder_inputs = keras.Input(shape=(max_len,), dtype='int32')
    #encoder_inputs = keras.Input(batch_input_shape=(params['batch_size'], max_len,), dtype='int32')

    emb = Embedding(input_dim=nb_tokens,
                    output_dim=token_dim,
                    mask_zero=True,
                    trainable=True)(encoder_inputs)

    x = layers.LSTM(units=token_dim*2, return_sequences=False)(emb)
    x = layers.Dense(latent_dim, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()

    """
    ## Build the decoder
    """
    #latent_inputs = keras.Input(batch_input_shape=(params['batch_size'], latent_dim,), batch_shape=params['batch_size'])
    latent_inputs = keras.Input(shape=(latent_dim,))

    x = RepeatVector(max_len)(latent_inputs)
    x = layers.LSTM(64, return_sequences=True)(x)
    decoder_outputs = TimeDistributed(layers.Dense(units=nb_tokens, activation='softmax'), name='output')(x)

    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam())

    decoder.summary()

    #(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    #mnist_digits = np.concatenate([x_train, x_test], axis=0)
    #mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

    vae = VAE(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Nadam())

    return vae


class SentEmbedder:
    def __init__(self):
        pass

    def load(self, model_dir):
        with open(os.path.join(model_dir, 'text_vae.config'), 'rb') as f:
            self.cfg = pickle.load(f)

        self.max_len = self.cfg['max_len']
        self.token2index = self.cfg['token2index']
        self.index2token = dict((i, t) for t, i in self.token2index.items())

        self.bpe_model = load_bpe_model(self.cfg['bpe_model_name'])

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


if __name__ == '__main__':
    tmp_dir = '../tmp'
    data_dir = '../data'
    save_dir = '../tmp/vae'

    params = dict()

    params['spm_items'] = 20000
    params['token_dim'] = 32
    params['latent_dim'] = 128
    params['batch_size'] = 256

    computed_params = dict()

    sents = load_sents()

    bpe_model_name = train_bpe_model(sents, params, save_dir)
    computed_params['bpe_model_name'] = bpe_model_name

    bpe_model = load_bpe_model(bpe_model_name)

    samples, X, samples_params = load_samples(sents, bpe_model, max_phrase_charlen=40)
    computed_params.update(samples_params)

    train_samples, val_samples, X_train, X_val = train_test_split(samples, X, test_size=1000)

    vae = create_model(params, computed_params)

    visualizer = VisualizeCallback(X_val, val_samples, vae, computed_params, save_dir)

    vae.fit(X_train,
            epochs=100,
            batch_size=params['batch_size'],
            callbacks=[visualizer],)

    with open(os.path.join(save_dir, 'text_vae.encoder.arch'), 'w') as f:
        f.write(vae.encoder.to_json())

    with open(os.path.join(save_dir, 'text_vae.decoder.arch'), 'w') as f:
        f.write(vae.decoder.to_json())

    with open(os.path.join(save_dir, 'text_vae.config'), 'wb') as f:
        pickle.dump({**params, **computed_params}, f)
