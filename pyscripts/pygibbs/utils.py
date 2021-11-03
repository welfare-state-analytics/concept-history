import numpy as np
import pandas as pd
from collections import Counter
from bidict import bidict
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess

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

def distinctWords(Nk, vocab, K, n=20, beta=0):
    """
    Computes top n p(z|w)
    beta is set to 0 if hyperparameter is symmetric (results are proportional.)
    """
    vocab_rev = bidict(vocab).inv
    word_list = []
    results = (Nk + beta) / (Nk + beta).sum(axis=0)[np.newaxis]

    for k in range(K):
        indices = (-results[k]).argsort()[:n]
        word_list.append([vocab_rev.get(key) for key in indices])

    distinctwords = pd.DataFrame(np.array(word_list).T)
    distinctwords.columns = list(range(K))

    return distinctwords

def topWords_old(phi, vocab, K, stopwords=False, n=20):
    """
    Computes top n p(w|K=k)
    :param stopwords: list of stopwords   
    """
    if stopwords != False:
        idx = [i for i in [vocab.get(word) for word in stopwords] if i is not None]
        phi = np.delete(phi, idx, axis=1)
    vocab_rev = bidict(vocab).inv
    word_list = []
    
    for k in range(K):
        indices = (-phi[k]).argsort()[:n]
        word_list.append([vocab_rev.get(key) for key in indices])
    topwords = pd.DataFrame(np.array(word_list).T)
    topwords.columns = list(range(K))

    return topwords

def topWords(phi, vocab, K, stopwords=False, n=20):
    """
    Computes top n p(w|K=k)
    :param stopwords: list of stopwords   
    """
    if stopwords != False:
        idx = [i for i in [vocab.get(word) for word in stopwords] if i is not None]
        vocab = {i:word for i,word in enumerate([word for word in vocab if word not in stopwords])}
        phi = np.delete(phi, idx, axis=1)

    word_list = []
    for k in range(K):
        indices = (-phi[k]).argsort()[:n]
        word_list.append([vocab.get(key) for key in indices])
    topwords = pd.DataFrame(np.array(word_list).T)
    topwords.columns = list(range(K))

    return topwords

def senses_over_time(time, doc, z, K, color, relative=True):
    """
    Plots senses over time from integer lists of inputs.
    Relative normalizes sense counts to proportions in each given timeframe.
    """
    z_long = [z[m] for m in doc]
    df = pd.DataFrame({'time': time, 'z': z_long})
    df = pd.crosstab(df["time"], df["z"], normalize='index' if relative else False)
    df["time"] = df.index
    df = pd.melt(df, 'time')

    fig, ax = plt.subplots()
    plt.xlabel('time')
    plt.ylabel('proportion' if relative else 'frequency')
    if relative:
        plt.yticks(np.arange(0, 1, 0.2))
    plt.legend([f'sense:{k}' for k in range(K)])
    ax.spines[["top","right"]].set_visible(False)

    pal = sns.color_palette(color)
    palmap = dict_variable = {key:value for (key,value) in zip(list(range(K)), pal)}
    ax = sns.lineplot(x=df['time'], y=df['value'], hue=df['z'], palette=palmap)

    return (fig, ax)

def word_freq_over_time(time, target, color, relative=True):
    """
    Plots word freqs over time from integer lists of inputs.
    Target word list should be lematized.
    """
    df = pd.DataFrame({'time': time, 'target': target})
    df = pd.crosstab(df["time"], df["target"])

    lemmas = sorted(list(set(target)))
    legend = []
    fig, ax = plt.subplots()

    if relative == True:
        df = df.div(df.sum(axis=1), axis=0)
        plt.yticks(np.arange(0, 1.01, 0.2))

    for i,lemma in enumerate(lemmas):
        plt.plot(df.index, df[lemma], color=color[i])
        legend.append(f'lemma: {lemma}')
    plt.legend(legend)
    plt.xlabel('time')
    plt.ylabel('proportion' if relative else 'frequency')
    ax.spines[["top","right"]].set_visible(False)

    return (fig, ax)

def z_sorter(z, phi, Nk, K):
    """
    Sorts z in descending frequency order for easier comparisons between runs and models.
    """
    z_sorted = [count[0] for count in Counter(z).most_common()]
    z_list = list(range(K))
    z_sorted.extend(z_list[len(z_sorted)-K:]) # adds back empty topics

    z = [z_list[z_sorted.index(zi)] for zi in z]
    phi = phi[z_sorted]
    Nk = Nk[z_sorted]
    return (z, phi, Nk)

def git_auto_push(path, message):
    """
    Used to automatically upload results inbetween training of models.
    """
    subprocess.call('git config --global credential.helper store', shell=True)
    subprocess.call('git pull', shell=True)
    subprocess.call(f'git add {path}', shell=True)
    subprocess.call(f'git commit -m "{message}"', shell=True)
    subprocess.call('git push', shell=True)

    