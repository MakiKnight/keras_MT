# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 09:29:25 2018

@author: julien.angeloni
"""

import string
import re
from pickle import dump, load
from unicodedata import normalize
from numpy import array
from numpy.random import rand
from numpy.random import shuffle

# On charge les données depuis le fichier
def load_doc(filename):
    # On ouvre le fichier en lecture
    file = open(filename, mode='rt', encoding='UTF-8')
    
    # On lit tout le texte
    text = file.read()
    
    # On ferme le fichier
    file.close()
    
    return text

# On sépare le texte pour avoir des "phrases"
def to_pairs(doc):
    lines = doc.strip().split('\n')
    pairs = [line.split('\t') for line in lines]
    return pairs

# On nettoie les lignes
def clean_pairs(lines):
    cleaned = list()
    # Création d'une regex pour filtrer les caractères
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    
    # Préparation de la table de traduction pour retirer la ponctuation
    table = str.maketrans('','',string.punctuation)
    
    for pair in lines:
        clean_pair = list()
        for line in pair:
            # Normalisation des caractères unicode
            line = normalize('NFD', line).encode('ascii', 'ignore')
            line = line.decode('UTF-8')
            
            # Codage des espaces
            line = line.split()
            
            # Codage en minuscule
            line = [word.lower() for word in line]
            
            # On retire la ponctuation
            line = [word.translate(table) for word in line]
            
            # On retire les caractères non "imprimables" des token
            line = [re_print.sub('',w) for w in line]
            
            # On retire les token avec des nombres
            line = [word for word in line if word.isalpha()]
            
            # On stock sous forme de String
            clean_pair.append(' '.join(line))
        cleaned.append(clean_pair)
    return array(cleaned)

# Pour sauvegarder les données nettoyées
def save_clean_data(sentences, filename):
    dump(sentences, open(filename, 'wb'))
    print('Saved: %s' % filename)

# Chargement des données
filename = 'deu.txt'
doc = load_doc(filename)

# Séparation en Englais-Allemand pair
pairs = to_pairs(doc)

# Nettoyage du contenu
clean_pairs = clean_pairs(pairs)

# Sauvegarde des paires nettoyées sous un fichier du nom 'english-german.pkl
save_clean_data(clean_pairs, 'english-german.pkl')

'''
Pour vérifier que ce que nous avons est correct.
# On vérifie
for i in range(100):
    print('[%s] => [%s]' % (clean_pairs[i,0], clean_pairs[i,1]))
'''

# Charger le contenu nettoyé
def load_clean_sentences(filename):
    return load(open(filename, 'rb'))

# On charge les données
raw_dataset = load_clean_sentences('english-german.pkl')

# Réduction de la taille du dataset
n_sentences = 10000
dataset = raw_dataset[:n_sentences, :]

# On mélange le tout
shuffle(dataset)

# On sépare le tout en train/test
train, test = dataset[:9000], dataset[9000:]

# On sauvegarde le tout
save_clean_data(dataset, 'english-german-both.pkl')
save_clean_data(train, 'english-german-train.pkl')
save_clean_data(test, 'english-german-test.pkl')