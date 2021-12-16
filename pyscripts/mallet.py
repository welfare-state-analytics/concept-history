import os, json
from pygibbs import utils
import numpy as np
import pandas as pd
import subprocess as sp
import gzip
import re

def mallet_dataset(mallet_path, input, results_path, keep_sequence=True):
	"""
	Creates a (training) dataset in .mallet format.
	"""
	cmd = [ mallet_path,
		"import-dir",
		"--input", input,
		"--output", os.path.join(results_path, 'train.mallet'),
		"--keep-sequence", str(keep_sequence),
		]
	proc = sp.Popen(cmd)
	proc.wait()
	print(f'Dataset stored in {results_path}.')

def _logperplex(mallet_output, print_out=True):
	"""
	Scrapes log perplexity from mallet console output.
	"""
	lp = []
	for line in str(mallet_output).split('\\n'):
		if print_out==True: 
			print(line.replace('\\t', ' '))
		if "LL/token" in line:
			lp.append(float(line.split()[-1].replace(',', '.')))
	return np.array(lp)

def _get_state(results_path):
	"""
	Reads mallet binary state file and creates a formatted dataset from it.
	"""
	with gzip.open(os.path.join(results_path, "state.gz"), "rb") as f:
		nrows = len(lines := f.readlines()[3:])
		variables = ["doc", "w", "z", "word", "idx", "file"]
		df = pd.DataFrame(np.zeros((nrows, len(variables))), columns=variables)
		df = df.astype(dtype={v:"int" if v != "file" and v != "word" else "str" for v in variables})
		for i, line in enumerate(lines):
			doc, file, idx, w, word, z = str(line).split()
			doc = int(re.findall('[0-9]+', doc)[0])
			z = int(re.findall('[0-9]+', z)[0])
			w = int(w)
			idx = int(idx)
			df.loc[i, variables] = doc, w, z, word, idx, file
	return df

def _get_results(results_path):
	"""
	Generates results (numpy objects and vocab) from state
	"""
	df = _get_state(results_path)

	# FOR TESTING
	#df["file"] = list(map(lambda x: x.split('.txt')[0] + '_5.txt', df["file"]))

	vocab = {}
	V = len(vocab)
	M = max(df["doc"])+1
	Nd = np.zeros((M, K), dtype=int)
	Nk = np.zeros((K, V), dtype=int)
	Z = np.zeros(M, dtype=int)
	for _, row in df.iterrows():
		m, w, z, word, idx, file = row[["doc", "w", "z", "word", "idx", "file"]]
		if word not in vocab:
			vocab[word] = w
		Nd[m,z] += 1
		Nk[z,w] += 1
		target_idx = int(re.findall('[0-9]+', file)[-1])
		if target_idx == idx:
			print(row["word"])
			Z[m] = z
	theta = Nd / Nd.sum(axis=1)[:,np.newaxis]
	phi = Nk / Nk.sum(axis=1)[:,np.newaxis]

	results = [theta, phi, Nd, Nk, Z, vocab]
	return results

def mallet_lda(mallet_path, results_path, K, alpha, beta, epochs, print_out=True):
	"""
	Main LDA function. Runs mallet train-topics and gathers results from final state.
	Returns: [theta, phi, Nd, Nk, Z, vocab, logperplexity]
	"""
	cmd = [ mallet_path,
			"train-topics",
			"--input", os.path.join(results_path, 'train.mallet'),
			"--output-model", os.path.join(results_path, "lda.model"),
			"--output-state", os.path.join(results_path, "state.gz"),
			"--num-topics", str(K),
			"--alpha", str(alpha*K),
			"--beta", str(beta),
			"--num-iterations", str(epochs),
			]

	proc = sp.Popen(cmd, stdout = sp.PIPE, stderr = sp.PIPE)
	proc.wait()
	print('model is finished at least ...')
	mallet_output = proc.stderr.read()
	lp = _logperplex(mallet_output, print_out=print_out)
	results = _get_results(results_path)
	results.append(lp)
	return results

# Run
mallet_path = os.path.expanduser('~/mallet-2.0.8/bin/mallet')
K, alpha, beta = 5, 0.5, 0.01
epochs = 100

# for dataset in
p, c, k = 'j', 5, 5
data_dir = f'data/mallet_windows/{p}_window_{c}'
#data_dir = os.path.expanduser('~/mallet-2.0.8/sample-data/web/en') # TEST FOR DEBUGGING

# Create output folder
subdirs = ['results/lda-models', p, f'window_{c}_topic_{k}', 'model']
for i in range(len(subdirs)):
	out = '/'.join(subdirs[:i+1])
	exec('try:os.mkdir(out)\nexcept:pass')

# Main LDA functions
mallet_dataset(mallet_path, data_dir, out)
theta, phi, Nd, Nk, Z, vocab, logperplexity = mallet_lda(
	mallet_path, out, K=K, alpha=alpha, beta=beta, epochs=0)

# Store results
np.save(os.path.join(out, 'theta.npy'), theta)
np.save(os.path.join(out, 'phi.npy'), phi)
np.save(os.path.join(out, 'Nd.npy'), Nd)
np.save(os.path.join(out, 'Nk.npy'), Nk)
np.save(os.path.join(out, 'z.npy'), Z)
np.save(os.path.join(out, 'logperplexity.npy'), logperplexity)
with open(os.path.join(out, 'vocab.json'), 'w') as f:
	json.dump(vocab, f, ensure_ascii=True, indent=4)

