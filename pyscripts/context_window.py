"""
Script for creating context window data.
To do: Currently repeats the process for each window size.
       Instead just do it for the largest window and extract smaller windows from it.
"""
import numpy as np
import os, json
import re
import progressbar
from pyparlaclarin.read import speeches_with_name
from lxml import etree

def count_files(path, extension='.xml'):
    """
    Counts number of files to be opened. Used for updating the progressbar.
    """
    n_files = 0
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                n_files += 1
    return n_files

def corpus_iterator(path):
    """
    Generator for paths to xml files in riksdagen corpus
    """
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.xml'):
                yield '/'.join([subdir, file])

def speech_processor(speech):
    """
    Creates list of lowercase words with special characters removed from speeech.
    """
    speech = re.sub(r'[^A-Za-zÀ-ÿ ]+', '', speech)
    speech = speech.lower()
    speech = speech.split()
    return speech

def create_context(speech, i, window_size):
    lb = max(0, i-window_size)
    ub = min(len(speech), i+window_size + 1)
    context = speech[lb:ub]
    return context

def create_contexts(crp, target, window_size, n_files):
    """
    : param crp: generator object for xml filepaths
    : param target: list of target words to create windows around
    : param window_size: number of word tokens on each side of target word
    : param n_files: 
    """
    keys = ["w", "doc", "target", "file_dir"]
    data = {key: [] for key in keys}
    n_pseudodocs, c = 0, 0

    with progressbar.ProgressBar(max_value=n_files) as bar:
        for file_path in crp:
            root = etree.parse(file_path, parser).getroot()
            file_dir = '/'.join(file_path.split('/')[3:5])

            for speech, idx in speeches_with_name(root, return_ids=True):
                speech = speech_processor(speech)

                if set(speech).intersection(target):
                    for i, word in enumerate(speech):
                        if word in target:
                            
                            # Create context windows and store additional information
                            context = create_context(speech, i, window_size)
                            Nm = len(context)
                            data["w"].extend(context)
                            data["doc"].extend([n_pseudodocs]*Nm)
                            data["target"].extend([word]*Nm)
                            data["file_dir"].extend([file_dir]*Nm)

                            n_pseudodocs += 1 # Increment doc count
            bar.update(c)
            c += 1
            
    return data

target_f = ['information', 'informations', 'informationen', 'informationens', 'informationer', 'informationers', 'informationerna', 'informationernas',\
            'upplysningar', 'upplysningars', 'upplysningarna', 'upplysningarnas', 'upplysning', 'upplysnings', 'upplysningen', 'upplysningens'\
            'propaganda', 'propagandas', 'propagandan', 'propagandans']

target_j = ['medium', 'media', 'medial', 'mediala', 'medialt', 'medias', 'medier', 'mediers', 'mediernas', 'mediet', 'mediets', 'massmedium', \
            'massmedia', 'massmedial', 'massmediala', 'massmedialt', 'massmedias', 'massmedier', 'massmediernas', 'massmediet', 'massmediets']

window_sizes = [5, 10, 50, 100]
projects = ['f', 'j']

in_path = '../riksdagen-corpus/corpus'
n_files = count_files(in_path)
parser = etree.XMLParser(remove_blank_text=True)

for project in projects:
    target = target_f if project == 'f' else target_j

    for window_size in window_sizes:
        out_path = 'data/context_windows'
        # Create output directories
        try:
            os.mkdir(out_path)
        except:
            pass
        crp = corpus_iterator(in_path)
        data = create_contexts(crp, target, window_size, n_files)

        file_name = f'{project}_window_{window_size}.json'
        with open('/'.join([out_path, file_name]), "w") as outfile:
            json.dump(data, outfile)
        print(f'Window size {window_size} finished for project {project}.')
