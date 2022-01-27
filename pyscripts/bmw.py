"""
TODO: Make redirected output also print.
"""
import numpy as np
import pandas as pd
import os, json
import time
from collections import Counter
import little_mallet_wrapper as lmw
import subprocess as sp
import gzip

def import_file(path_to_mallet, training_data,
				path_to_training_data, output, **kwargs):
	"""
	Modified lmw function to create mallet format dataset taking same args as java verison.
	"""

	with open(path_to_training_data, 'w') as training_data_file:
		for i, d in enumerate(training_data):
			training_data_file.write(str(i) + ' no_label ' + d + '\n')

	# Create subprocess arguments
	cmd = 	[path_to_mallet, "import-file",
			"--input", path_to_training_data,
			"--output", path_to_formatted_training_data,
			"--keep-sequence"]

	for key, value in kwargs.items():
		cmd.extend([f"--{key.replace('_', '-')}", str(value)])

	proc = sp.call(cmd)
	print('Complete')

def train_topics(path_to_mallet, input_path, **kwargs):
	"""
	Mallet wrapper taking same args as java version with few differences.
		args:
		- path_to_mallet, path to mallet binaries
		- input_path, same as --input in mallet
		kwargs: 
		- all other train-topics arguments in mallet
		- underscore is used instead of hyphens everywhere
		- additional argument logperplexity returns list of LL/token values if set to anything
	"""
	start = time.time()

	logperplexity=False
	if 'logperplexity' in kwargs:
		kwargs.pop('logperplexity')
		logperplexity = []

	# Create subprocess arguments
	cmd = 	[path_to_mallet, "train-topics",
			"--input", input_path]
	for key, value in kwargs.items():
		cmd.extend([f"--{key.replace('_', '-')}", str(value)])

	
	# Run process and redirect output
	with sp.Popen(cmd, stderr=sp.PIPE, bufsize=1, universal_newlines=True) as p:
		for line in p.stderr:
			print(line, end='') # process line here
			if "LL/token" in line and logperplexity != False:
				logperplexity.append(float(line.split()[-1].replace(',', '.')))

	if logperplexity != False:
		return logperplexity

def train_topic_model(path_to_mallet,
                      path_to_formatted_training_data,
                      logperplexity=False,
                      **kwargs):

    cmd =   [path_to_mallet, 'train-topics',
            '--input', path_to_formatted_training_data,]

    for key, value in kwargs.items():
        cmd.extend([f"--{key.replace('_', '-')}", str(value)])

    lp = []
    print('Training topic model...')
    with sp.Popen(cmd, stderr=sp.PIPE, bufsize=1, universal_newlines=True) as p:
        for line in p.stderr:
            print(line, end='')
            if "LL/token" in line and logperplexity == True:
                lp.append(float(line.split()[-1].replace(',', '.')))

    print('Complete')
    if logperplexity == True:
        return lp

def get_state(output_state):
	"""
	Reads mallet binary state file and creates a formatted dataset from it.
	args:
	- print_ab prints alpha and beta
	TODO: Found existing functions for getting state, check if quicker
	https://jeriwieringa.com/2018/07/17/pyLDAviz-and-Mallet/
	"""
	start = time.time()
	state = []
	f = gzip.open(output_state, "rb")
	
	for i, line in enumerate(f):
		if i >= 3:
			s = line.decode().split()
			doc = int(s[0].split("b'")[-1])
			w = int(s[3])
			z = int(s[5].split('\\n')[0])
			word = s[4]
			state.append([doc, w, z, word])
		#else:
		#	print(line.decode())

	f.close()
	state = pd.DataFrame(state, columns=['doc', 'w', 'z', 'word'])
	#print(f'State retrieved after {round(time.time()-start)} seconds.')
	return state

def get_results(state, pos=None):
	"""
	Retrieve model parameters and topic indicator counts from state dataframe.
	Position returns topic indicators only for words in pos positions (count matrices are the same).
	"""
	start = time.time()
	vocab = {}
	M = max(state["doc"])+1
	V = len(set(state["w"]))
	K = len(set(state["z"]))
	Nd = np.zeros((M, K), dtype=int)
	Nk = np.zeros((K, V), dtype=int)
	Z = np.zeros(M, dtype=int) if pos else []
	for m in range(M):
		doc = state.loc[state["doc"] == m]
		if pos:
			Z[m] = doc.iloc[pos[m]]["z"]
		for i,row in doc.iterrows():
			doc, w, z, word = row[["doc", "w", "z", "word"]]
			Nd[doc,z] += 1
			Nk[z,w] += 1
			if pos == None:
				Z.append(z)
			if word not in vocab:
				vocab[word] = w
			
	theta = Nd / Nd.sum(axis=1)[:,np.newaxis]
	phi = Nk / Nk.sum(axis=1)[:,np.newaxis]
	results = [theta, phi, Nd, Nk, Z]
	print(f'Results retrieved after {round(time.time()-start)} seconds.')
	return [results, vocab]

# Globals
thresh = 5
path_to_mallet = os.path.expanduser('~/mallet-2.0.8/bin/mallet')
path_to_data = 'data/context_windows'

for p in ['j', 'f']:
	for c in [10]:
		for k in [20, 50]:
			start = time.time()
			path_to_results = f'results/lda-models/{p}/window_{c}_topic_{k}/model'
			path_to_training_data           = path_to_results + '/training.txt'
			path_to_formatted_training_data = path_to_results + '/mallet.training'
			path_to_model                   = path_to_results + '/mallet.model'
			path_to_state                   = path_to_results + '/state.gz'

			# Load data
			df = pd.read_csv(os.path.join(path_to_data, f'{p}_window_{c}.csv'))
			with open('data/target-words.json', 'r') as f:
				target_words = json.load(f)["target_" + p]

			# Preprocessing
			M = max(df["doc"]) + 1
			word_counts = Counter(' '.join(df["w"]).split())
			for m, row in df.iterrows():
				words = []
				for i, word in enumerate(row["w"].split()):
					if word_counts.get(word) > thresh or word in target_words:
						words.append(word)
					elif i < row["pos"]:
						df.loc[m, "pos"] -= 1
				df.loc[m,"w"] = ' '.join(words)
				if len(words) == 0:
					print(f'Document {i} is empty after filtering rare words and is removed.')
					df = df.drop(i)
			df.reset_index(drop=True)
			df["doc"] = df.index
			year = np.array(df["year"])
			target = np.array(df["target"])
			training_data = df["w"].tolist()

			# Run LDA
			import_file(path_to_mallet=path_to_mallet,
						training_data=training_data,
						path_to_training_data=path_to_training_data,
						output=path_to_formatted_training_data,
						token_regex=r'\S+',
						)

			lp = train_topic_model(path_to_mallet=path_to_mallet,
								path_to_formatted_training_data=path_to_formatted_training_data,
								logperplexity=True,
								output_model=path_to_model,
								output_state=path_to_state,
								num_iterations=10000,
								num_topics=k,
								alpha=0.5*k,
								beta=0.5,
								)

			state = get_state(output_state=path_to_state)
			print(f'State retrieved after {np.round(time.time()-start, 3)} seconds')
			
			results, vocab = get_results(state, df["pos"].tolist())
			theta, phi, Nd, Nk, z = results

			# Store results
			np.save(os.path.join(path_to_results, 'logperplexity.npy'), lp)
			np.save(os.path.join(path_to_results, 'theta.npy'), theta)
			np.save(os.path.join(path_to_results, 'phi.npy'), phi)
			np.save(os.path.join(path_to_results, 'Nd.npy'), Nd)
			np.save(os.path.join(path_to_results, 'Nk.npy'), Nk)
			np.save(os.path.join(path_to_results, 'z.npy'), z)
			np.save(os.path.join(path_to_results, 'year.npy'), year)
			np.save(os.path.join(path_to_results, 'target.npy'), target)
			with open(os.path.join(path_to_results, 'vocab.json'), 'w') as f:
				json.dump(vocab, f)
