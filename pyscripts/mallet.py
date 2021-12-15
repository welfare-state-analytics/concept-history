from gensim.models.wrappers import ldamallet
import os, json
from pygibbs import utils
import gensim
from gensim import corpora
import numpy as np
import pandas as pd
import subprocess as sp

mallet_path = os.path.expanduser('~/mallet-2.0.8/bin/mallet')
data_dir = os.path.expanduser('~/mallet-2.0.8/sample-data/web/en')

K = 5
cmd = [ mallet_path,
		"import-dir",
		"--input", data_dir,
		"--output", "train.mallet",
		"--keep-sequence", "TRUE",
		]
proc = sp.Popen(cmd)
proc.wait()

cmd = [ mallet_path,
		"train-topics",
		"--input", "train.mallet",
		"--num-topics", str(K),
		"--alpha", "0.5",
		"--beta", "0.5",
		"--num-iterations", "1000",
		"--output-doc-topics", "theta.txt",
		"--topic-word-weights-file", "phi.txt", # Nk+beta
		# --word-topic-counts-file FILENAME potentially interesting
		]

proc = sp.Popen(cmd, stdout = sp.PIPE, stderr = sp.PIPE)
proc.wait()
output = proc.stderr.read()

logperplex = []
for line in str(output).split('\\n'):
	if "LL/token" in line:
		logperplex.append(float(line.split()[-1].replace(',', '.')))

theta = []
with open('theta.txt', 'r') as f:
	for line in f:
		theta.append(list(map(lambda x: float(x), line.split("\t")[2:])))
theta = np.array(theta)

with open('phi.txt', 'r') as f:
	df = pd.DataFrame(list(map(lambda x: x.split(), f.readlines())), columns=["z", "w", "count"])
vocab = {key:value for value,key in enumerate(sorted(set(df["w"])))}
phi = np.zeros((K, len(vocab)))
for _,row in df.iterrows():
	phi[int(row["z"]), vocab.get(row["w"])] = row["count"]
phi = phi / phi.sum(axis=1)[:,np.newaxis]

# Looks good
# Add infer_z
# Add assymetric beta?

#print(df.loc[df["w"] == target])
#print(vocab)
# sp.call([mallet_path, "train-topics", "--info"]) # info
#
#with open('data/context_windows/j_window_5.json') as f:
#	data = json.load(f)
#
#w = data["w"]
#doc = data["doc"]
#target = data["target"]
#M = max(doc)
#
## Preprocessing
#doc, w = utils.rareWords(doc, w, target, thresh=10)
#w = np.array(w)
#doc = np.array(doc)
#target = np.array(target)
#doc_tokenized = [list(w[doc == m]) for m in range(M)]
#
## Mallet
#K = 5
#alpha = 0.5
#dct = corpora.Dictionary()
#corpus = [dct.doc2bow(doc, allow_update=True) for doc in doc_tokenized]
#id_words = [[(dct[id], count) for id, count in line] for line in corpus]



#subprocess.call([mallet_path, "info"])
#~/mallet-2.0.8/bin/mallet info

#proc = subprocess.Popen([mallet_path, "info"], stdout=subprocess.PIPE)
#output = proc.stdout.read()

# Seems to work in capturing the output



#command = 'pyscripts/run-mallet.py'
#p = subprocess.Popen(
#        [command],
#        shell=True,
#        stdin=None,
#        stdout=subprocess.PIPE,
#        stderr=subprocess.PIPE,
#        close_fds=True)
#out, err = p.communicate()
#print(out)
#print(err)

#p = sp.Popen(["pyscripts/run-mallet.py"], stdout=sp.PIPE, stderr=sp.PIPE)
#out, err = p.communicate()
#result = p.returncode
#print(result)
#model = ldamallet.LdaMallet(mallet_path, corpus=corpus, num_topics=K, id2word=dct, alpha=alpha, iterations = 20)
	




#model.save('ldamodel.model')
#model = ldamallet.LdaMallet.load('lda_model.model')
#model = ldamallet.malletmodel2ldamodel(model)

# theta
def infer_parameters(model, corpus, M, K):
	phi = np.exp(model.state.get_Elogbeta()) # Elogbeta is log(phi)
	theta = np.zeros((M, K))
	for m in range(M):
		for k, prob in model[corpus[m]]:
			theta[m,k] = prob
	return (theta, phi)

#theta, phi = infer_parameters(model, corpus, M, K)

def infer_z(theta, phi, M, dictionary):
	z = np.zeros(M, dtype=int) 
	for m in range(M):
		wi = dct.token2id[target[m]]
		probs = theta[m] * phi[:,wi]
		probs = probs / probs.sum()
		z[m] = np.argmax(probs)
	return z

#Nk = model.state.get_lambda()

#print(model.log_perplexity(corpus))

#print(model.get_term_topics())
#print(model.get_term_topics().shape)
#print(model.get_topic_terms())
#print(model.get_topics())







p = 'j'
c = 5
k = 5
out = f'lda-models/{p}/window_{c}_topic_{k}/model'



