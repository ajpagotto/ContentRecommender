import numpy as np
import re

def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan



def convert_int(x):
    try:
        return int(x)
    except:
        return np.nan

def tokenize(text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokenized = tokenizer.tokenize(text)
    mod = [w.lower() for w in tokenized if len(w) > 1 and w.lower() not in stopwords.words('english')]
    words = [w for w in mod if not w.isnumeric()]
    return  words


def calc_cosine_similarity(v1, v2):
    return 1 - spatial.distance.cosine(v1, v2)