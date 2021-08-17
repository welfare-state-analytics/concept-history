from typing import Counter
import numpy as np
import pandas as pd
from collections import deque
import os,json
import re
from progressbar import progressbar
from pyparlaclarin.read import speeches_with_name
from lxml import etree

# Target words
target = ['Information', 'information', 'informations', 'informationen', 'informationens', 'informationer', 'informationers', 'informationerna', 'informationernas',\
'Upplysning', 'upplysningar', 'upplysningars', 'upplysningarna', 'upplysningarnas', 'upplysning', 'upplysnings', 'upplysningen', 'upplysningens'\
'Propaganda', 'propaganda', 'propagandas', 'propagandan', 'propagandans']

# Context window size (on each side of target word)
window = 50
data_list = []

# Take subset for debugging (set to 0 for all files)
max_folders = 0
c = 0

riksdagen_path = '/home/robinsaberi/Git_Repos/riksdagen-corpus/corpus'

# Extract number of folders
n_folders = 0
for path, subdirs, files in os.walk(riksdagen_path):
    for name in subdirs:
        n_folders += 1

# Traverse corpus folder and subdirectories
#for root, dirs, files in os.walk(riksdagen_path):
#    path = root.split(os.sep)
#    folder = os.path.basename(root)
#    if folder.isnumeric():
#        for file in progressbar(files):
#            file_path = '/'.join([riksdagen_path,folder,file])
#            parser = etree.XMLParser(remove_blank_text=True)
#            root = etree.parse(file_path, parser).getroot()
#
#            # Iterate over speeches within xml file
#            for speech in speeches_with_name(root):
#
#                # Process string
#                speech = re.sub(r'[^A-Za-z0-9À-ÿ /-]+', '', speech) # Remove special characters
#                speech = speech.lower()
#                txt = deque(speech.split()) # Not sure if needed, but easy way to pad beginning of list
#
#                # Pad documents in case context window goes outside of paragraph
#                txt.extendleft(['']*window)
#                txt.extend(['']*window)
#                txt = list(txt)
#
#                # Find target word indices
#                pos = [i for i,word in enumerate(txt) if word in target]
#
#                # Retrieve pseudo-documents
#                for i in pos:
#                    context = txt[(i-window):(i+window+1)]
#                    context.append('/'.join([folder, file]))
#                    data_list.append(context)
#
#        # Overall progress
#        c += 1
#        if c % 5 == 0:
#            print(f'{round(c / n_folders * 100, 2)}% of folders finished')
#        
#        # For debugging
#        if c == max_folders:
#            break
#        
## Write as df for exploration
#df = pd.DataFrame(data_list)
#df.rename(columns={ df.columns[-1]: "folder/file" }, inplace = True)
#df.to_csv("test_context_window_data.csv", index=False)

### Initialize and reformat as json ###
df = pd.read_csv('test_context_window_data.csv')
df = df.fillna('')
df = df.drop('folder/file', 1)


# Extract document id and words to list format

#doc,w = [word for sentence in enumerate(df) for word in sentence if word != '']
#print(w)


#w_counts = Counter({})
#[w_counts.update(word) for word in df.values]
#print(w_counts)
#
#doc,w,n = [],[],[]
#for i, row in df.iterrows():
#    words = row[row != '']
#    w.extend(words)
#    n.append(len(words))
#    doc.extend([i]*len(words))
#print(w)
#
#vocab = set(w)
#vocab = {token:idx+1 for idx, token in enumerate(vocab)}
#w = [vocab[token] -1 for token in w]

V = len(vocab)
M = max(doc)+1
N = len(w)

data = {
    "V": V,
    "M": M,
    "N": N,
    "w": w,
    "doc": doc,
    "n": n
}

# Make objects json compatible
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
with open("test_context_window_data.json", "w") as outfile: 
    json.dump(data, outfile, cls=NpEncoder)

with open("test_context_window_vocab.json", "w") as outfile: 
    json.dump(vocab, outfile, cls=NpEncoder)