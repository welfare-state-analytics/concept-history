"""
Script for creating context window data.
To do: Currently repeats the process for each window size.
       Instead just do it for the largest window and extract smaller windows from it.
"""
import numpy as np
import os, json
import re
from lxml import etree
import yaml
import argparse
from time import time

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

def speech_iterator(root):
    """
    Convert Parla-Clarin XML to an iterator of of concatenated speeches and speaker ids.
    Speech segments are concatenated unless a new speaker appears (ignoring any notes).

    Args:
        root: Parla-Clarin document root, as an lxml tree root.
    """
    us = root.findall('.//{http://www.tei-c.org/ns/1.0}u')
    if len(us) == 0: return None
    idx_old = us[0].attrib.get("who", "")
    speech = []

    for u in us:
        for text in u.itertext():
            idx = u.attrib.get("who", "")
            if idx != idx_old:
                yield([' '.join(speech), idx])
                speech = []
            speech.extend(text.split())
            idx_old = idx

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

def create_contexts(crp, target, window_size, n_files, parser):
    """
    : param crp: generator object for xml filepaths
    : param target: list of target words to create windows around
    : param window_size: number of word tokens on each side of target word
    : param n_files: 
    """
    keys = ["w", "doc", "target", "file_path", "id", "pos"]
    data = {key: [] for key in keys}
    n_pseudodocs = 0

    for file_path in crp:
        root = etree.parse(file_path, parser).getroot()
        speeches = speech_iterator(root)
        if speeches != None:
            for speech, idx in speeches:
                speech = speech_processor(speech)
                if set(speech).intersection(target):
                    for i, word in enumerate(speech):
                        if word in target:
                            context = create_context(speech, i, window_size)
                            Nm = len(context)
                            data["w"].extend(context)
                            data["doc"].extend([n_pseudodocs]*Nm)
                            data["target"].append(word)
                            data["file_path"].append(file_path)
                            data["id"].append(idx)
                            data["pos"].append(i)
                            n_pseudodocs += 1
            
    return data

def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    with open(config["target"]) as f:
        targets = json.load(f)
    parser = etree.XMLParser(remove_blank_text=True)
    n_files = count_files(config["corpus"])

    for window_size in config["window_sizes"]:
        start = time()
        for project in config["projects"]:
            crp = corpus_iterator(config["corpus"])
            data = create_contexts(crp, targets["target_"+project], window_size, n_files, parser)
            file_name = f'{project}_window_{window_size}.json'
            with open('/'.join([config["data"], file_name]), "w") as outfile:
                json.dump(data, outfile)
            print(f'Window size {window_size} finished for project {project} after {(time()-start)//60} minutes.')

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("--config", type=str)
    args = argparser.parse_args()
    main(args)