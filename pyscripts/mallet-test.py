import numpy as np
import pandas as pd
import os, json
from pygibbs import utils
import matplotlib.pyplot as plt

path = 'results/lda-models'

project = 'j'
c = 5
k = 5

results = os.path.join(path, project, f'window_{str(c)}_topic_{str(k)}')

logperplexity = np.load(f'{results}/model/logperplexity.npy')
theta = np.load(f'{results}/model/theta.npy')
phi = np.load(f'{results}/model/phi.npy')
Nd = np.load(f'{results}/model/Nd.npy')
Nk = np.load(f'{results}/model/Nk.npy')
z = np.load(f'{results}/model/z.npy')
years = np.load(f'{results}/model/time.npy')
target = np.load(f'{results}/model/target.npy')

with open(f'{results}/model/vocab.json', 'r') as f:
	vocab = json.load(f)
with open('data/stopwords-sv.txt','r') as f:
    stopwords = f.read().splitlines()

color = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1',\
		 '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF'][:k]

out = os.path.join(results, 'analysis')
exec('try:os.mkdir(out)\nexcept:pass')

f = utils.convergence_plot(logperplexity, color=color, xticks=np.arange(0, 10000, 10))
plt.savefig(os.path.join(out, 'logperplexity.png'), dpi=300, bbox_inches='tight')
plt.close('all')

# years is broken atm
f = utils.senses_over_time(years, z, color)
plt.savefig(os.path.join(out, 'senses_over_time.png'), dpi=300, bbox_inches='tight')
#plt.show()
plt.close('all')

#f = utils.word_freq_over_time(years, lemmas, color)
#plt.savefig(os.path.join(out, 'word_prop_over_time.png'), dpi=300, bbox_inches='tight')
#plt.close('all')
print(target)
topwords = utils.topWords(phi, vocab, k, stopwords, list(set(target)))
topwords.to_csv(os.path.join(out, 'top_words.csv'), index=False)
print(topwords)

distinctwords = utils.distinctWords(Nk, vocab, k)
distinctwords.to_csv(os.path.join(out, 'distinct_words.csv'), index=False)
print(distinctwords)

#topdocs = utils.topDocs(doc, w, posterior, k, file_dir, n=20, seed=123)
#with open('/'.join([out, 'top_docs_by_topic.txt']), 'w') as f:
#    for line in topdocs:
#        f.write(line + '\n'*2)


