import numpy as np
import pandas as pd
import os, json
from pygibbs import utils
import matplotlib.pyplot as plt
import argparse
import yaml
from time import time

# Cleans years of form 197677 to 1977
def cleanYear(year):
    if len(year) == 4:
        return int(year)
    return int(year[:4])

def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    results = config["results"]
    projects = config["projects"]
    window_sizes = config["window_sizes"]
    K = config["K"]

    for project in projects:
        for c in window_sizes:
            for k in K:
                start = time()
                in_path = f'{results}/{project}/window_{c}_topic_{k}/model'
                Nd = np.load(os.path.join(in_path, 'Nd.npy'))
                Nk = np.load(os.path.join(in_path, 'Nk.npy'))
                theta = np.load(os.path.join(in_path, 'theta.npy'))
                phi = np.load(os.path.join(in_path, 'phi.npy'))
                z = np.load(os.path.join(in_path, 'z.npy'))
                logdensity = np.load(os.path.join(in_path, 'logdensity.npy'))
                posterior = np.load(os.path.join(in_path, 'posterior.npy'))

                meta = pd.read_csv(f'{config["data"]}/{project}_meta.csv')

                with open(os.path.join(config["corpus"], 'party_mapping.json'), 'r') as f:
                    party_map = json.load(f)
                with open(os.path.join(config["data"], f'{project}_window_{c}.json'), 'r') as f:
                    data = json.load(f)
                with open(os.path.join(in_path, 'vocab.json'), 'r') as f:
                    vocab = json.load(f)
                with open(config["stopwords"],'r') as f:
                    stopwords = f.read().splitlines()

                doc = data["doc"]
                w = data["w"]
                file_dir = data["file_path"]
                target = data["target"]
                
                years = list(map(lambda x: cleanYear(x.split('/')[3]), file_dir))
                lemmas = list(map(lambda x: x[:4], data["target"])) # Lemmatize
                M, V = len(z), np.shape(Nk)[1]
                z, phi, Nk = utils.z_sorter(z, phi, Nk, k)

                color = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1',\
                         '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF'][:k]

                out = os.path.join(results, project, f'window_{c}_topic_{k}', 'analysis')
                try:
                    os.mkdir(out)
                except:
                    pass

                f = utils.convergence_plot(logdensity, config["burn_in"], color)
                plt.savefig(os.path.join(out, 'logdensity.png'), dpi=300, bbox_inches='tight')
                plt.close('all')

                f = utils.senses_over_time(years, doc, z, color)
                plt.savefig(os.path.join(out, 'senses_over_time.png'), dpi=300, bbox_inches='tight')
                plt.close('all')

                try:
                    os.mkdir(os.path.join(out, 'gender'))
                except:
                    pass

                for gender in ["woman", "man"]:
                    f = utils.senses_over_time(years, doc, z, color, meta, "gender", gender)
                    f.suptitle('Women' if gender == "woman" else "Men")
                    gender_abbrev = "m" if gender == "man" else "f"
                    plt.savefig(os.path.join(out, f'gender/senses_over_time_{gender_abbrev}.png'), dpi=300, bbox_inches='tight')

                try:
                    os.mkdir(os.path.join(out, 'party'))
                except:
                    pass

                parties = [p for p in list(set(meta["party_abbrev"])) if str(p) != 'nan']
                for party in parties:
                    f = utils.senses_over_time(years, doc, z, color, meta, "party_abbrev", party, xlim=False)
                    f.suptitle(max([key for key,value in party_map.items() if value == party]))
                    plt.savefig(os.path.join(out, f'party/senses_over_time_{party}.png'), dpi=300, bbox_inches='tight')

                f = utils.word_freq_over_time(years, lemmas, color)
                plt.savefig(os.path.join(out, 'word_prop_over_time.png'), dpi=300, bbox_inches='tight')
                plt.close('all')

                topwords = utils.topWords(phi, vocab, k, stopwords, list(set(target)))
                topwords.to_csv(os.path.join(out, 'top_words.csv'), index=False)
                
                distinctwords = utils.distinctWords(Nk, vocab, k)
                distinctwords.to_csv(os.path.join(out, 'distinct_words.csv'), index=False)

                topdocs = utils.topDocs(doc, w, posterior, k, file_dir, n=20, seed=123)
                with open('/'.join([out, 'top_docs_by_topic.txt']), 'w') as f:
                    for line in topdocs:
                        f.write(line + '\n'*2)

                print(f'Project "{project}" with window size {c} finished after {(time()-start)//60} minutes.')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("--config", type=str)
    args = argparser.parse_args()
    main(args)