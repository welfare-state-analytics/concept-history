import numpy as np
import pandas as pd
from collections import deque
import os,json
import re
from progressbar import progressbar
from utils import paragraph_iterator
from lxml import etree
from itertools import chain

# Constructs dataset for word sense induction
# Requires pyriksdagen folders in cd
# Should change generator, can get documents like "interpellation om forskning undervisning och information på näringslärans område" multiple times
# Only slight preprocessing of text is implemented atm

# Target words
target = ['Information', 'information', 'informations', 'informationen', 'informationens', 'informationer', 'informationers', 'informationerna', 'informationernas',\
'Upplysning', 'upplysningar', 'upplysningars', 'upplysningarna', 'upplysningarnas', 'upplysning', 'upplysnings', 'upplysningen', 'upplysningens'\
'Propaganda', 'propaganda', 'propagandas', 'propagandan', 'propagandans']

# Context window size (on each side of target word)
window = 25
data_list = []

# Take subset for debugging (set to 0 for all files)
max_folders = 0
c = 0

riksdagen_path = '/home/robinsaberi/Git_Repos/riksdagen-corpus/corpus'

# Traverse corpus folder and subdirectories
#for root, dirs, files in os.walk("corpus/"):
for root, dirs, files in os.walk(riksdagen_path):
    path = root.split(os.sep)
    folder = os.path.basename(root)
    if folder.isnumeric():
        for file in progressbar(files):
            file_path = '/'.join([riksdagen_path,folder,file])
            parser = etree.XMLParser(remove_blank_text=True)
            root = etree.parse(file_path, parser).getroot()

            # Create generator for paragraphs within xml file
            p_iter = paragraph_iterator(root)

            for p in p_iter:
                # Process string
                p = re.sub(r'[^A-Za-z0-9À-ÿ /-]+', '', p) # Remove special characters
                p = p.lower()
                txt = deque(p.split()) # Not sure if needed, but easy way to pad beginning of list

                # Pad documents in case context window goes outside of paragraph
                txt.extendleft(['']*window)
                txt.extend(['']*window)
                txt = list(txt)

                # Find target word indices
                pos = [i for i,word in enumerate(txt) if word in target]

                # Retrive pseudo-documents
                for i in pos:
                    context = txt[(i-window):(i+window+1)]
                    context.append(file_path)
                    data_list.append(context)

        print(f'Folder {folder} finished')

        # For debugging
        c += 1
        if c == max_folders:
            break

# Write as df for exploration
df = pd.DataFrame(data_list)
df.rename(columns={ df.columns[-1]: "file" }, inplace = True)
df.to_csv("test_context_window_data.csv", index=False)

### Initialize and reformat as json ###

# Creates list of length N with document ids
doc = list(chain(*([x]*(len(df.columns)-1) for x in range(len(df)))))

# Create vocabulary and word tokens
w = [word for sentence in df.drop("file", axis=1).values for word in sentence]
vocab = set(w)-{'NA'}
vocab = {token:idx+1 for idx, token in enumerate(vocab)}
w = [vocab[token] -1 if str(token) != 'nan' else np.nan for token in w]
V = len(vocab)
M = max(doc)+1
N = len([token for token in w if str(token) != 'nan']) # Excluding padded values

# Hyperparams
K = 5
alpha = [0.5]*K
beta = [0.5]*V

# Initialize z
z = (np.random.choice(K, M))

data = {
    "K": K,
    "V": V,
    "M": M,
    "N": N,
    "alpha": alpha,
    "beta": beta,
    "z": z,
    "w": w,
    "doc": doc
}

# Make objects json compatible
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

# Write file
with open("test_context_window_data.json", "w") as outfile: 
    json.dump(data, outfile, cls=NpEncoder)