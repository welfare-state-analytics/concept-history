import numpy as np
import pandas as pd
from collections import Counter
from bidict import bidict
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
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

def distinctWords(Nk, vocab, n=20):
    """
    Computes top n p(z|w)
    """
    vocab_rev = bidict(vocab).inv

    K = len(Nk)
    word_list = []

    # results = (Nk+beta) / (Nk+beta).sum(axis=0)[np.newaxis]
    # Can exclude beta from equation as it does not contribute to the order of distinct words
    results = Nk / Nk.sum(axis=0)[np.newaxis]

    for k in range(K):
        indices = (-results[k]).argsort()[:n]
        word_list.append([vocab_rev.get(key) for key in indices])

    distinctwords = pd.DataFrame(np.array(word_list).T)
    distinctwords.columns = list(range(K))

    return distinctwords

def topWords(phi, vocab, stopwords=False, n=20):
    """
    Computes top n p(w|K=k)
    :param stopwords: list of stopwords   
    """
    if stopwords != False:
        idx = [value for value in [vocab.get(word) for word in stopwords] if value is not None]
        phi = np.delete(phi, idx, axis=1)

    vocab_rev = bidict(vocab).inv

    K = len(phi)
    word_list = []

    for k in range(K):
        indices = (-phi[k]).argsort()[:n]
        word_list.append([vocab_rev.get(key) for key in indices])

    topwords = pd.DataFrame(np.array(word_list).T)
    topwords.columns = list(range(K))

    return topwords

def time_sense_plot_old(time, z, M, K, relative=False):
    """
    Depreciated version used for test data.
    Plots senses over time from integer lists of inputs.
    Relative normalizes sense counts to proportions in each given timeframe.
    """
    time_unique = sorted(list(set(time)))

    # Crosstabulate sense by year
    df = pd.DataFrame(np.zeros(shape=(len(time_unique), K), dtype=np.int32),\
                               columns=list(range(K)), index=time_unique)
    for m in range(M):
        df.loc[time[m], z[m]] += 1

    if relative == True:
        df = Normalizer(norm='l1').fit_transform(df)

    # Plot attributes can further be modified from return object
    fig, ax = plt.subplots()
    plt.plot(df)
    plt.xlabel('time')
    plt.ylabel('proportion' if relative else 'frequency')
    plt.yticks(np.arange(0, 1, 0.2))
    plt.legend([f'sense:{k}' for k in range(K)])
    ax.spines[["top","right"]].set_visible(False)
    return fig, ax

def senses_over_time(time, doc, z, K, relative=True):
    """
    Plots senses over time from integer lists of inputs.
    Relative normalizes sense counts to proportions in each given timeframe.
    """
    z_long = [z[doc[i]] for i in range(len(doc))]
    df = pd.DataFrame({'time': time, 'z': z_long})
    df = pd.crosstab(df["time"], df["z"])

    if relative == True:
        df = df.div(df.sum(axis=1), axis=0)

    fig, ax = plt.subplots()
    plt.plot(sorted(set(time)), df)
    plt.xlabel('time')
    plt.ylabel('proportion' if relative else 'frequency')
    if relative == True:
        plt.yticks(np.arange(0, 1.01, 0.2))
    plt.legend([f'sense:{k}' for k in range(K)])
    ax.spines[["top","right"]].set_visible(False)

    return (fig, ax)

def word_freq_over_time(time, target, relative=True):
    """
    Plots word freqs over time from integer lists of inputs.
    Target word list should be lematized.
    """
    #if len(set(target)) == 1:
    #    target = list(target)

    df = pd.DataFrame({'time': time, 'freq': target})
    df = pd.crosstab(df["time"], df["freq"])

    if relative == True:
        df = df.div(df.sum(axis=1), axis=0)

    fig, ax = plt.subplots()
    plt.plot(sorted(set(time)), df)
    plt.xlabel('time')
    plt.ylabel('frequency')
    if relative == True:
        plt.yticks(np.arange(0, 1.01, 0.2))

    plt.legend([f'word:{v}' for v in sorted(list(set(target)))])
    ax.spines[["top","right"]].set_visible(False)

    return (fig, ax)

def z_sorter_old(z, K):
    """
    Sorts z in descending frequency order for easier comparisons between runs and models.
    """
    z_sorted = [count[0] for count in Counter(z).most_common()]
    z_new = list(range(K))
    return [z_new[z_sorted.index(zi)] for zi in z]

def z_sorter(z, phi, Nk, K):
    """
    Sorts z in descending frequency order for easier comparisons between runs and models.
    """
    z_sorted = [count[0] for count in Counter(z).most_common()]
    z_list = list(range(K))
    z = [z_list[z_sorted.index(zi)] for zi in z]

    # Sort phi, Nk accordingly
    #idx = np.empty_like(z_sorted)
    #idx[z_sorted] = np.arange(len(z_sorted))
    #phi = phi[:, idx]
    #Nk = Nk[:, idx]
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