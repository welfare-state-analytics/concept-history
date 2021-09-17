import numpy as np
import pandas as pd
from collections import Counter
from bidict import bidict

def rareWords(doc, w, thresh=5):
    """
    Drops rare words given lists of equal lenghts with documents and word tokens
    """
    if len(doc) != len(w):
        print('Documents and word tokens are not of equal length')
        return

    c = Counter(w)
    rare_words = set([key for key, value in c.items() if value < thresh])

    idx= []
    for i in range(len(w)):
        if w[i] not in rare_words:
            idx.append(i)

    w = [w[i] for i in idx]
    doc = [doc[i] for i in idx]

    return (doc, w)

def tokenize(w):
    """
    Maps words to integers and returns these together with vocabulary
    """
    vocab = set(w)
    vocab = {token:i for i, token in enumerate(vocab)}
    w = [vocab[token] for token in w]
    return w, vocab

def initDataframe(doc, w, M):
    """
    Creates dataframe used in Gibbs sampler from lists
    """
    df = pd.DataFrame(np.array((doc,w)).T)
    df.columns = ["doc", "w"]
    df = df.groupby('doc')['w'].apply(list).reset_index()
    return df

def distinctWords(Nk, beta, vocab, n=10):
    """
    Computes top n p(z=k|w)
    """
    vocab_rev = bidict(vocab).inv

    K = len(Nk)
    word_list = []
    results = (Nk+beta) / (Nk+beta).sum(axis=0)[np.newaxis]

    for k in range(K):
        indices = (-results[k]).argsort()[:n]
        word_list.append([vocab_rev.get(key) for key in indices])

    return np.array(word_list).T
