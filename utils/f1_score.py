import re
import numpy as np
from sklearn import metrics


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = {'a','an','the'}


def word_extraction(sentence):   
    words = re.sub("[^\w]", " ",  sentence).split()    
    return words


def tokenize(sentences):

    words = []
    for sentence in sentences:
        w = word_extraction(sentence)
        words.extend(w)
    words = sorted(list(set(words)))   
    return words


def prep(text):
    
    text = text.replace('\n', ' ').lower()              # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ',text)            # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('',text)                  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join([w for w in text.split() if not w in STOPWORDS])# delete stopwords from text
    return text


def calculate_f1(w1, w2):
    
    vocab = tokenize([w1, w2])
    true_words = word_extraction(w1)
    predicted_words = word_extraction(w2)
    true_bag_vector = np.zeros(len(vocab))
    predicted_bag_vector = np.zeros(len(vocab))
    
    for w in true_words:
        for i, word in enumerate(vocab):
            if word == w:
                true_bag_vector[i] += 1
                
    for w in predicted_words:
        for i, word in enumerate(vocab):
            if word == w:
                predicted_bag_vector[i] += 1
                
    macro_f1 = metrics.f1_score(true_bag_vector, predicted_bag_vector, average = 'macro')
    
    return macro_f1