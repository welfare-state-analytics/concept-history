import numpy as np
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

def main(folder):
    in_path = os.path.join(folder, 'model')
    Nd = np.load(os.path.join(in_path, 'Nd.npy'))
    Nk = np.load(os.path.join(in_path, 'Nk.npy'))
    theta = np.load(os.path.join(in_path, 'theta.npy'))
    phi = np.load(os.path.join(in_path, 'phi.npy'))
    z = np.load(os.path.join(in_path, 'z.npy'))
    logdensity = np.load(os.path.join(in_path, 'logdensity.npy'))
    logposterior = np.load(os.path.join(in_path, 'logposterior.npy'))

    file = '_'.join(folder.split('/')[-1].split('_')[:3]) + '.json'
    
    with open(os.path.join('data/context_windows', file)) as f:
        data = json.load(f)
    with open(os.path.join(in_path, 'vocab.json')) as f:
        vocab = json.load(f)
    with open('data/stopwords-sv.txt','r') as f:
        stopwords = f.read().splitlines()

    doc = data["doc"]
    w = data["w"]
    file_dir = data["file_path"]
    years = list(map(lambda x: cleanYear(x.split('/')[3]), file_dir))
    target = list(map(lambda x: x[:4], data["target"])) # Lemmatize
    M, V, K = len(z), np.shape(Nk)[1], len(Nd)
    z, phi, Nk = utils.z_sorter(z, phi, Nk, K)

    color = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1',\
             '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF'][:K]

    out_path = os.path.join(folder, 'analysis')
    f = utils.convergence_plot(logdensity, 1000, color)
    plt.savefig(os.path.join(out_path, 'logdensity.png'), dpi=300, bbox_inches='tight')
    plt.close()

    f = utils.senses_over_time(years, doc, z, color)
    plt.savefig(os.path.join(out_path, 'senses_over_time.png'), dpi=300, bbox_inches='tight')
    plt.close()

    f = utils.word_freq_over_time(years, target, color)
    plt.savefig(os.path.join(out_path, 'word_prop_over_time.png'), dpi=300, bbox_inches='tight')
    plt.close()

    topwords = utils.topWords(phi, vocab, K, stopwords)
    topwords.to_csv(os.path.join(out_path, 'top_words.csv'), index=False)
    
    distinctwords = utils.distinctWords(Nk, vocab, K)
    distinctwords.to_csv(os.path.join(out_path, 'distinct_words.csv'), index=False)

    topdocs = utils.topDocs(doc, w, logposterior, K, file_dir, n=20, seed=123)
    with open('/'.join([out_path, 'top_docs_by_topic.txt']), 'w') as f:
        for line in topdocs:
            f.write(line + '\n'*2)

    print(f'Folder {folder} finished.')

if __name__ == "__main__":
    results = 'results/second-run'
    folders = sorted(os.listdir(results))
    folders = [f for f in folders if f.startswith('j')]
    for folder in folders:
        if os.path.isdir(os.path.join(results, folder)):
            main(os.path.join(results, folder))
