import numpy as np
import pandas as pd
import random
from collections import Counter
from bidict import bidict
import matplotlib.pyplot as plt
import seaborn as sns

def rareWords(doc, w, target, thresh=5):
    """
    Drops rare words given lists of equal lenghts with documents and word tokens
    """
    if len(doc) != len(w):
        print('Documents and word tokens are not of equal length')
        return
    c = Counter(w)
    rare_words = set([key for key, value in c.items() if value < thresh and key not in target])

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
    vocab = {token:i for i, token in enumerate(sorted(vocab))}
    w = [vocab[token] for token in w]

    return (w, vocab)

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

# add target words to be appended to stopwords
def topWords(phi, vocab, K, stopwords=None, targets=None, n=20):
    """
    Computes top n p(w|K=k)
    :param stopwords: list of stopwords
    """
    if stopwords and targets:
        stopwords.extend(targets)
    elif targets:
        stopwords = targets

    if stopwords:
        idx = [i for i in [vocab.get(word) for word in stopwords] if i is not None]
        vocab = {word:i for i,word in enumerate([word for word in vocab if word not in stopwords])}
        phi = np.delete(phi, idx, axis=1)

    word_list = []
    for k in range(K):
        indices = (-phi[k]).argsort()[:n]
        
        word_list.append([key for key,value in vocab.items() if value in indices])
    topwords = pd.DataFrame(np.array(word_list).T)
    topwords.columns = list(range(K))

    return topwords

def topDocs(doc, w, posterior, K, file_dir, n=10, seed=123):
    """
    Computes the top n p(m|K=k)
    Retrives documents most associated with each topic.
    Shuffling is used to solve conflicts with equally dominant documents.
    """
    random.seed(seed)
    M = len(posterior)
    indices_map = random.sample(range(M), M)
    indices_map = np.array([list(range(M)), indices_map]).T
    topdocs = []

    for k in range(K):
        # Get original topdoc indices from shuffled posterior
        indices = (-posterior[indices_map[:, 1], k]).argsort()[:n]
        indices = indices_map[indices, 0]
        topdocs.append(f'Topic {k}:')

        for idx in indices:
            word_indices = [i for i,m in enumerate(doc) if m == idx]
            words = ' '.join(w[word_indices[0]:word_indices[-1]+1])            
            topdocs.append(f'{words} ({file_dir[doc[word_indices[0]]]})')
    
    return topdocs

def lda_top_docs(df, theta, n=20, seed=123):
    M, K = np.shape(theta)
    np.random.seed(seed)
    np.random.shuffle(idx := list(range(M)))
    df_shuffled = df.loc[idx].reset_index(drop=True)
    topdocs = []
    for k in range(K):
        indices = (-theta[idx, k]).argsort()[:n]
        topdocs.append(f'Topic {k}:')
        for i in indices:
            file = '/'.join(df_shuffled.loc[i, "file_path"].split('/')[-2:])
            topdocs.append(f'{df_shuffled.loc[i, "w"]} ({file})')
    return topdocs

def topic_props_over_time_table(time, z, periods, meta=None, variable=None, value=None, relative=True):
    """
    Table with word proportions normalized over time periods.
    Args: periods is an array with periods as rows.
    """
    K = max(z)+1
    X = pd.DataFrame({'time': time, 'z': z})

    if isinstance(meta, pd.DataFrame) and variable and value:
        X[variable] = meta[variable]
        X = X.loc[X[variable] == value]
        X = X.reset_index(drop=True)

    df = pd.crosstab(X["time"], X["z"])
    results = pd.DataFrame(np.zeros((len(periods), K), dtype=int))
    for i, row in df.iterrows():
        idx = np.where(periods == i)[0][0]
        for k in range(K):
            try:
                results.loc[idx,k] += row[k]
            except:
                pass
    results.index = [str(periods[i, 0]) + '-' + str(periods[i,-1]) for i in range(len(periods))]
    results.loc['Total'] = results.sum(axis=0)
    if relative:
        results = results.div(results.sum(axis=1), axis=0)
    return np.round(results, 3)

def lemma_props_by_topic(lemmas, z, relative=True):
    X = pd.DataFrame({'lemmas':lemmas, 'z':z})
    X = pd.crosstab(X["lemmas"], X["z"], normalize='index' if relative else False)
    return np.round(X, 3)

def convergence_plot(loss, color=None, xticks=None):
    epochs = len(loss)
    df = pd.DataFrame({
        'epoch': list(range(epochs)) if len(xticks)==None else xticks,
        'y': loss})

    if color != None:
        pal = sns.color_palette(color)
    with sns.axes_style("whitegrid"):
        if color != None:
            f = sns.lineplot(x='epoch', y='y', data=df, palette=pal)
        else:
            f = sns.lineplot(x='epoch', y='y', data=df)
        #f.get_legend().remove()
        f.set_ylabel('')
        f.set(xlabel = "Epoch", ylabel = "Log joint density")
        sns.despine()

    return f

def senses_over_time(time, z, color, meta=None, variable=None, value=None, xlim=False):
    """
    Plots senses over time from integer lists of inputs.
    Relative normalizes sense counts to proportions in each given timeframe.
    """
    K = max(z)+1
    X = pd.DataFrame({'time': time, 'z': z})
    if isinstance(meta, pd.DataFrame) and variable and value:
        X[variable] = meta[variable]
        X = X.loc[X[variable] == value]
        X = X.reset_index(drop=True)

    f = plt.figure()
    gs = f.add_gridspec(2, 2)
    if color != None:
        pal = sns.color_palette(color)[:K]
        palmap = {key:value for (key,value) in enumerate(pal)}

    for col in range(gs.ncols):
        relative = True if col == 1 else False
        ylab = 'Proportion' if relative else 'Frequency'

        for row in range(gs.nrows):
            df = pd.crosstab(X["time"], X["z"], normalize='index' if relative else False)
            if row == 1: df = df.cumsum(axis=1)
            df["time"] = df.index
            df = pd.melt(df, 'time')

            with sns.axes_style("whitegrid"):
                ax = f.add_subplot(gs[row, col])
                sns.despine()
                if color != None:
                    g = sns.lineplot(x=df['time'], y=df['value'], hue=df['z'], palette=palmap)
                else:
                    g = sns.lineplot(x=df['time'], y=df['value'], hue=df['z'])
                g.get_legend().remove()
                if xlim == True:
                    g.set_xlim([min(time), max(time)])

                if row == 0:
                    g.set(xlabel=None)
                    g.set_ylabel(ylab)
                if row == 1:
                    g.set_ylabel(' '.join(['Stacked', ylab.lower()]))
                    for k in range(K):
                        lower = df.loc[df["z"] == k-1, "value"] if k != 0 else 0
                        upper = df.loc[df["z"] == k, "value"]
                        
                        if len(upper) == 0:
                            break
                        if color != None:
                            g.fill_between(df.loc[df["z"] == k, "time"], lower, upper, color=palmap[k])
                        else:
                            g.fill_between(df.loc[df["z"] == k, "time"], lower, upper)

    f.legend(labels=list(range(K)),
               loc="center right",
               title="Sense",
               bbox_to_anchor=(1, 0.5))

    return f

def word_freq_over_time(time, target, color):
    """
    Plots word freqs over time from integer lists of inputs.
    Target word list should be lematized.
    """
    lemmas = sorted(list(set(target)))
    df = pd.DataFrame({'time': time, 'target': target})
    df = pd.crosstab(df["time"], df["target"])
    df["time"] = df.index
    df = pd.melt(df, 'time')
    
    f = plt.figure()
    gs = f.add_gridspec(2, 1)

    if color != None:
        pal = sns.color_palette(color)[:len(lemmas)]
        palmap = {key:value for (key,value) in zip(lemmas,pal)}
    
    for row in range(2):
        if row == 1: df["value"] = df.groupby('time')["value"].transform(lambda x: x / x.sum())
        with sns.axes_style("whitegrid"):
            ax = f.add_subplot(gs[row])
            sns.despine()
            if color != None:
                g = sns.lineplot(x=df['time'], y=df['value'], hue=df['target'], palette=palmap)
            else:
                g = sns.lineplot(x=df['time'], y=df['value'], hue=df['target'])
            g.get_legend().remove()

            if row == 0:
                g.set(xlabel=None)
                g.set(ylabel="Frequency")

            if row == 1:
                g.set(xlabel="Time")
                g.set(ylabel="Proportion")
            
    f.legend(labels=lemmas,
           loc="center right",
           title="Lemmas",
           bbox_to_anchor=(1, 0.5))

    return f

def lemmas_over_time_by_topic(lemma, year, z, k):
    """
    TODO: Probably weight by overall word usage
    """
    X = pd.DataFrame({'lemma':lemma, 'year':year, 'z':z})
    Y = X.loc[X["z"] == k][["lemma", "year"]]
    Y = pd.crosstab(Y["year"], Y["lemma"])#, normalize='index')
    Y["year"] = Y.index
    with sns.axes_style("whitegrid"):
        for l in set(X["lemma"]):
            f = sns.lineplot(x=Y['year'], y=Y[l])
        f.set_title(f'Topic: {k}')
        f.set(xlabel='Year')
        f.set(ylabel='Frequency')
    sns.despine()
    return f

def z_sorter(z, theta, phi, Nd, Nk):
    """
    Sorts z in descending frequency order for easier comparisons between runs and models.
    """
    K = len(phi)
    z_sorted = [count[0] for count in Counter(z).most_common()]
    z_list = list(range(K))
    z_sorted.extend([k for k in range(K) if k not in z_sorted]) # adds back empty topics
    z = [z_list[z_sorted.index(zi)] for zi in z]
    theta = theta[:,z_sorted]
    phi = phi[z_sorted]
    Nd = Nd[:,z_sorted]
    Nk = Nk[z_sorted]
    return z, theta, phi, Nd, Nk, z_sorted



