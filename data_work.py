# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 10:14:53 2018

@author: julien.angeloni
"""

from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint

def load_clean_sentences(filename):
	return load(open(filename, 'rb'))

# On charge les données
dataset = load_clean_sentences('english-german-both.pkl')
train = load_clean_sentences('english-german-train.pkl')
test = load_clean_sentences('english-german-test.pkl')

# Utilisation d'un Tokenizer pour maper les mots à des entiers
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# On cherche la phrase la plus longue du dataset
def max_length(lines):
    return max(len(line.split()) for line in lines)

# On prépare le token pour les mots anglais
eng_tokenizer = create_tokenizer(dataset[:, 0])
eng_vocab_size =  len(eng_tokenizer.word_index) + 1
eng_length = max_length(dataset[:, 0])

print('English vocabulary size: %d' % eng_vocab_size)
print('English max length: %d' % eng_length)

# On fait de même pour les mots allemands
ger_tokenizer = create_tokenizer(dataset[:, 1])
ger_vocab_size = len(ger_tokenizer.word_index) +1
ger_length = max_length(dataset[:, 1])

print('German vocabulary size: %d' % ger_vocab_size)
print('German max length: %d' % ger_length)

# On encode et compense les sequences
def encode_sequences(tokenizer, length, lines):
    # En nombres entiers
    X = tokenizer.texts_to_sequences(lines)
    # On compense les séquences avec 0 valeur
    X = pad_sequences(X, maxlen=length, padding = 'post')
    return X

# On encode les sorties
def encode_output(sequences, vocab_size):
    ylist = list()
    for sequence in sequences:
        encoded = to_categorical(sequence, num_classes = vocab_size)
        ylist.append(encoded)
    y = array(ylist)
    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
    return y

# On prépare les données d'entrainement
trainX = encode_sequences(ger_tokenizer, ger_length, train[:, 1])
trainY = encode_sequences(eng_tokenizer, eng_length, train[:, 0])
trainY = encode_output(trainY, eng_vocab_size)

# On prépare les données de validation
testX = encode_sequences(ger_tokenizer, ger_length, test[:, 1])
testY = encode_sequences(eng_tokenizer, eng_length, test[:, 0])
testY = encode_output(testY, eng_vocab_size)

# On définit la Neural Machine Translation
def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
    model = Sequential()
    model.add(Embedding(src_vocab, n_units, input_length = src_timesteps, mask_zero = True))
    model.add(LSTM(n_units))
    model.add(RepeatVector(tar_timesteps))
    model.add(LSTM(n_units, return_sequences = True))
    model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
    return model

# Definition du modele
model = define_model(ger_vocab_size, eng_vocab_size, ger_length, eng_length, 256)
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy')

# On affiche des informations sur le modele
print(model.summary())
#plot_model(model, to_file = 'model.png', show_shapes = True)

# On adapte le modele
filename = 'model.h5'
checkpoint = ModelCheckpoint(filename, monitor = 'val_loss', verbose = 1, save_best_only = True, mode='min')
model.fit(trainX, trainY, epochs=30, batch_size=64, validation_data=(testX, testY), callbacks=[checkpoint], verbose=2)