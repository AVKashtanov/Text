import pandas as pd
import pickle
import numpy as np
import re
import tflearn

from nltk.stem.snowball import RussianStemmer
from collections import Counter
from nltk.tokenize import TweetTokenizer

tweets_col_number = 3
VOCAB_SIZE = 1000

print("Начинаем забирать модель")
net = tflearn.input_data([None, VOCAB_SIZE])
net = tflearn.fully_connected(net, 125, activation='ReLU')
net = tflearn.fully_connected(net, 25, activation='ReLU')
net = tflearn.fully_connected(net, 2, activation='softmax')
model = tflearn.DNN(net)
model.load('model/model.tfl')
print("Забрали модель")

with open('token/tokenizer.pickle', 'rb') as handle:
    vocab = pickle.load(handle)
print("Забрали токены")

stem_count = Counter()
tokenizer = TweetTokenizer()

stemer = RussianStemmer()
regex = re.compile('[^а-яА-Я ]')
stem_cache = {}


def get_stem(token):
    stem = stem_cache.get(token, None)
    if stem:
        return stem
    token = regex.sub('', token).lower()
    stem = stemer.stem(token)
    stem_cache[token] = stem
    return stem

token_2_idx = {vocab[i] : i for i in range(VOCAB_SIZE)}


def tweet_to_vector(tweet, show_unknowns=False):
    vector = np.zeros(VOCAB_SIZE, dtype=np.int_)
    for token in tokenizer.tokenize(tweet):
        stem = get_stem(token)
        idx = token_2_idx.get(stem, None)
        if idx is not None:
            vector[idx] = 1
        elif show_unknowns:
            print("Unknown token: {}".format(token))
    return vector




def test_tweet(tweet):
    tweet_vector = tweet_to_vector(tweet, True)
    positive_prob = model.predict([tweet_vector])[0][1]
    print('Комментарий: {}'.format(tweet))
    print('Вероятность позитивности = {:.5f}. Результат: '.format(positive_prob),
          'Позитивный' if positive_prob > 0.5 else 'Негативный')
    print('-------------')
    return inputdef()

def inputdef():
    tweet = input("Введите отзыв: ")
    return test_tweet(tweet)

inputdef()
