import numpy as np
import pandas as pd
import os, json
from pygibbs import utils
import matplotlib.pyplot as plt

# Cleans years of form 197677 to 1977
def cleanYear(year):
    if len(year) == 4:
        return int(year)

    year = list(year)
    year = ''.join(year[0:2] + year[4:6])
    return int(year)

folders = sorted(os.listdir('results'))

for folder in folders:
    results = os.path.join('results', folder)

    Nd = np.load(os.path.join(results, 'Nd.npy'))
    Nk = np.load(os.path.join(results, 'Nk.npy'))
    theta = np.load(os.path.join(results, 'theta.npy'))
    phi = np.load(os.path.join(results, 'phi.npy'))
    z = np.load(os.path.join(results, 'z.npy'))

    # Dictionary and stopwords
    with open(os.path.join(results, 'vocab.json')) as f:
        vocab = json.load(f)
    stopwords = open('data/stopwords-sv.txt','r')
    stopwords = stopwords.read().splitlines()

    # Results
    distinctwords = utils.distinctWords(Nk, vocab, 20)
    topwords = utils.topWords(phi, vocab, stopwords, 20)

    # Import target word and year from context window data
    file = '_'.join(folder.split('_')[:3]) + '.json'
    with open(os.path.join('data/context_windows', file)) as f:
        data = json.load(f)
    
    doc = data["doc"]
    years = list(map(lambda x: cleanYear(x.split('/')[0]), data["file_dir"]))

    M, V, K = len(z), np.shape(Nk)[1], len(Nd)

    # Sort topic indicators in descending order
    z, phi, Nk = utils.z_sorter(z, phi, Nk, K)

    fig, ax = utils.senses_over_time(years, doc, z, K)
    plt.savefig(os.path.join(results, 'senses_over_time.png'), dpi=300, bbox_inches='tight')

    # Plot wordfreqs differently as there are different numbers of target words
    if folder.startswith('f'):
        target = list(map(lambda x: x[:4], data["target"]))
        fig, ax = utils.word_freq_over_time(years, target)
        plt.savefig(os.path.join(results, 'word_prop_over_time.png'), dpi=300, bbox_inches='tight')
        plt.close()

    elif folder.startswith('j'):
        target = ['media']*len(data["target"])

    fig, ax = utils.word_freq_over_time(years, target, relative=False)
    plt.savefig(os.path.join(results, 'word_freq_over_time.png'), dpi=300, bbox_inches='tight')
    plt.close()

    topwords = utils.topWords(phi, vocab, stopwords)
    distinctwords = utils.distinctWords(Nk, vocab)

    topwords.to_csv(os.path.join(results, 'top_words.csv'), index=False)
    distinctwords.to_csv(os.path.join(results, 'distinct_words.csv'), index=False)
    print(f'Folder {folder} finished.')