# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 11:42:27 2018

@author: julien.angeloni
"""

from pickle import load
from numpy import array
from numpy import argmax
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu

# Charger un dataset nettoyé
def load_clean_sentences(filename):
	return load(open(filename, 'rb'))
 
# Lier un tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer
 
# Récuperer la taille max
def max_length(lines):
	return max(len(line.split()) for line in lines)
 
# Encoder et decoder
def encode_sequences(tokenizer, length, lines):
	# Encode en entier
	X = tokenizer.texts_to_sequences(lines)
	# on tamponne les vlaurs à 0
	X = pad_sequences(X, maxlen=length, padding='post')
	return X
 
# On mappe un entier à un mot
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None
 
# On génère une cible selon la source
def predict_sequence(model, tokenizer, source):
	prediction = model.predict(source, verbose=0)[0]
	integers = [argmax(vector) for vector in prediction]
	target = list()
	for i in integers:
		word = word_for_id(i, tokenizer)
		if word is None:
			break
		target.append(word)
	return ' '.join(target)
 
# On évalue le modèle
def evaluate_model(model, tokenizer, sources, raw_dataset):
	actual, predicted = list(), list()
	for i, source in enumerate(sources):
		# On traduit le code source embedded
		source = source.reshape((1, source.shape[0]))
		translation = predict_sequence(model, eng_tokenizer, source)
		raw_target, raw_src = raw_dataset[i]
		if i < 10:
			print('src=[%s], target=[%s], predicted=[%s]' % (raw_src, raw_target, translation))
		actual.append(raw_target.split())
		predicted.append(translation.split())
	# On score avec la méthode BLEU
	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))
 
# On charge le dataset
dataset = load_clean_sentences('english-german-both.pkl')
train = load_clean_sentences('english-german-train.pkl')
test = load_clean_sentences('english-german-test.pkl')

# On prépare le token anglais
eng_tokenizer = create_tokenizer(dataset[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(dataset[:, 0])

# Idem pour l'allemand
ger_tokenizer = create_tokenizer(dataset[:, 1])
ger_vocab_size = len(ger_tokenizer.word_index) + 1
ger_length = max_length(dataset[:, 1])

# On prépare les données
trainX = encode_sequences(ger_tokenizer, ger_length, train[:, 1])
testX = encode_sequences(ger_tokenizer, ger_length, test[:, 1])
 
# On charge le modele
model = load_model('model.h5')
translation = predict_sequence(model, eng_tokenizer, encode_sequences(ger_tokenizer, ger_length, 'ich bin ein berliner'))
print(translation)
'''
# On le teste
print('train')
evaluate_model(model, eng_tokenizer, trainX, train)

# On le test sur des séquences
print('test')
evaluate_model(model, eng_tokenizer, testX, test)
'''