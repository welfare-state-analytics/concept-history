def run_mallet():
	path_to_mallet_binary = os.path.expanduser('~/mallet-2.0.8/bin/mallet')

	with open('data/context_windows/j_window_5.json') as f:
		data = json.load(f)

	w = data["w"]
	doc = data["doc"]
	target = data["target"]
	M = max(doc)

	# Preprocessing
	doc, w = utils.rareWords(doc, w, target, thresh=10)
	w = np.array(w)
	doc = np.array(doc)
	target = np.array(target)
	doc_tokenized = [list(w[doc == m]) for m in range(M)]

	# Mallet
	K = 5
	alpha = 0.5
	dct = corpora.Dictionary()
	corpus = [dct.doc2bow(doc, allow_update=True) for doc in doc_tokenized]
	id_words = [[(dct[id], count) for id, count in line] for line in corpus]

	model = ldamallet.LdaMallet(path_to_mallet_binary, corpus=corpus, num_topics=K, id2word=dct, alpha=alpha, iterations = 20)
	model.save('ldamodel-test.model')


    #args = read_args()
print(1)
    #sys.exit(run_mallet())