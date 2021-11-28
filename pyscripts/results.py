import numpy as np
import pandas as pd
import os, json
from pygibbs import utils
import matplotlib.pyplot as plt

# Cleans years of form 197677 to 1977
def cleanYear(year):
    if len(year) == 4:
        return int(year)
    return int(year[:4])

def main():
    results = 'results/second-run'
    folders = [f for f in os.listdir(results) if os.path.isdir(os.path.join(results, f))]

    for folder in folders:
        in_path = os.path.join(results, folder, 'model')
        Nd = np.load(os.path.join(in_path, 'Nd.npy'))
        Nk = np.load(os.path.join(in_path, 'Nk.npy'))
        theta = np.load(os.path.join(in_path, 'theta.npy'))
        phi = np.load(os.path.join(in_path, 'phi.npy'))
        z = np.load(os.path.join(in_path, 'z.npy'))
        logdensity = np.load(os.path.join(in_path, 'logdensity.npy'))
        posterior = np.load(os.path.join(in_path, 'posterior.npy'))

        file = '_'.join(folder.split('/')[-1].split('_')[:3]) + '.json'
        
        meta = pd.read_csv(f'data/context_windows/{folder[0]}_meta.csv')
        with open('../riksdagen-corpus/corpus/party_mapping.json') as f:
            party_map = json.load(f)
        with open(os.path.join('data/context_windows', file)) as f:
            data = json.load(f)
        with open(os.path.join(in_path, 'vocab.json')) as f:
            vocab = json.load(f)
        with open('data/stopwords-sv.txt','r') as f:
            stopwords = f.read().splitlines()

        doc = data["doc"]
        w = data["w"]
        file_dir = data["file_path"]
        target = data["target"]
        years = list(map(lambda x: cleanYear(x.split('/')[3]), file_dir))
        lemmas = list(map(lambda x: x[:4], data["target"])) # Lemmatize
        M, V, K = len(z), np.shape(Nk)[1], len(Nd)
        z, phi, Nk = utils.z_sorter(z, phi, Nk, K)

        color = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1',\
                 '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF'][:K]

        out_path = os.path.join(results, folder, 'analysis')
        f = utils.convergence_plot(logdensity, 1000, color)
        plt.savefig(os.path.join(out_path, 'logdensity.png'), dpi=300, bbox_inches='tight')
        plt.close()

        f = utils.senses_over_time(years, doc, z, color)
        plt.savefig(os.path.join(out_path, 'senses_over_time.png'), dpi=300, bbox_inches='tight')
        plt.close()

        for gender in ["woman", "man"]:
            f = utils.senses_over_time(years, doc, z, color, meta, "gender", gender)
            f.suptitle('Women' if gender == "woman" else "Men")
            gender_abbrev = "m" if gender == "man" else "f"
            plt.savefig(os.path.join(out_path, f'gender/senses_over_time_{gender_abbrev}.png'), dpi=300, bbox_inches='tight')

        parties = [p for p in list(set(meta["party_abbrev"])) if str(p) != 'nan']
        for party in parties:
            f = utils.senses_over_time(years, doc, z, color, meta, "party_abbrev", party)
            f.suptitle(max([key for key,value in party_map.items() if value == party]))
            plt.savefig(os.path.join(out_path, f'party/senses_over_time_{party}.png'), dpi=300, bbox_inches='tight')

        f = utils.word_freq_over_time(years, lemmas, color)
        plt.savefig(os.path.join(out_path, 'word_prop_over_time.png'), dpi=300, bbox_inches='tight')
        plt.close()

        topwords = utils.topWords(phi, vocab, K, stopwords, list(set(target)))
        topwords.to_csv(os.path.join(out_path, 'top_words.csv'), index=False)
        
        distinctwords = utils.distinctWords(Nk, vocab, K)
        distinctwords.to_csv(os.path.join(out_path, 'distinct_words.csv'), index=False)

        topdocs = utils.topDocs(doc, w, posterior, K, file_dir, n=20, seed=123)
        with open('/'.join([out_path, 'top_docs_by_topic.txt']), 'w') as f:
            for line in topdocs:
                f.write(line + '\n'*2)

        print(f'Folder {folder} finished.')

if __name__ == "__main__":
    main()
