import numpy as np
import os,json
import re
from numpy.core.defchararray import index
import progressbar
from pyparlaclarin.read import speeches_with_name
from lxml import etree

# Target words
target_f = ['Information', 'information', 'informations', 'informationen', 'informationens', 'informationer', 'informationers', 'informationerna', 'informationernas',\
'Upplysning', 'upplysningar', 'upplysningars', 'upplysningarna', 'upplysningarnas', 'upplysning', 'upplysnings', 'upplysningen', 'upplysningens'\
'Propaganda', 'propaganda', 'propagandas', 'propagandan', 'propagandans']

target_j = ['medium', 'media', 'medial', 'mediala', 'medialt', 'medias', 'medier', 'mediers', 'mediernas', 'mediet', 'mediets', 'massmedium', \
          'massmedia', 'massmedial', 'massmediala', 'massmedialt', 'massmedias', 'massmedier', 'massmediernas', 'massmediet', 'massmediets']

targets = [target_f, target_j]

# Context window size (on each side of target word)
windows = [5,10,50,100]
riksdagen_path = 'corpus'

# Extract number of folders (for progressbar)
n_files = 0
for path, subdirs, files in os.walk(riksdagen_path):
    for folder in subdirs:
        for file in folder:
            n_files += 1

# Make objects json compatible
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)

for target in targets:
    doc_folder = {} # dict for document-file mappings
    project = 'f'
    
    # For each window size
    for window in windows:
        with progressbar.ProgressBar(max_value=n_files) as bar: # Bar is broken ¯\_(ツ)_/¯
    
            # Initialize
            keys = ["w", "doc"]
            data = {key: [] for key in keys}
            n_pseudodocs, c = 0,0
    
            # Traverse corpus subdirs and files
            for root, dirs, files in os.walk(riksdagen_path):
                path = root.split(os.sep)
                folder = os.path.basename(root)
                if folder.isnumeric():
                    for file in files:
                        if file.endswith('.xml'):
                            file_path = '/'.join([riksdagen_path,folder,file])
                            parser = etree.XMLParser(remove_blank_text=True)
                            root = etree.parse(file_path, parser).getroot()
        
                            # Iterate over speeches within xml file
                            for speech,idx in speeches_with_name(root, return_ids=True):
        
                                # Process string
                                speech = re.sub(r'[^A-Za-zÀ-ÿ ]+', '', speech) # Remove special characters
                                speech = speech.lower()
                                par = speech.split()

                                # Find target words in paragraph
                                if set(par).intersection(target):
                                    for i,word in enumerate(par):
                                        if word in target:
                                            
                                            # Create context window and store w,doc
                                            lb = max(0, i-window)
                                            ub = min(len(par), i+window + 1)
                                            ctx = par[lb:ub]
                                            data["w"].extend(ctx)
                                            data["doc"].extend([n_pseudodocs]*len(ctx))
        
                                            # Store folder/file only for 1 window
                                            if window == min(windows):
                                                folder_key = '/'.join([folder,file])
                                                if folder_key not in doc_folder:
                                                    doc_folder[folder_key] = []
        
                                                doc_folder[folder_key].append('_'.join([str(n_pseudodocs),idx]))
                                            
                                            n_pseudodocs += 1 # Increment doc count
        
                            bar.update(c)
                
        with open(f"ctx_window_data/ctx_window_{window}_{project}.json", "w") as outfile: 
            json.dump(data, outfile, cls=NpEncoder)
    
    with open(f"ctx_window_data/doc_folder_mapping_{project}.json", "w") as outfile: 
            json.dump(doc_folder, outfile, cls=NpEncoder)
    
    project = 'j'
    