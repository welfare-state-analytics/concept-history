"""
Script for creating context window data.
"""
import pandas as pd
import os, json
import re
from lxml import etree
import yaml
import argparse
from time import time
 
def cleanYear(year):
   if len(year) == 4:
       return int(year)
   return int(year[:4])
 
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
   pos = i - lb
   return [context, pos]

def create_contexts(crp, target, window_size, n_files, parser):
   """
   : param crp: generator object for xml filepaths
   : param target: list of target words to create windows around
   : param window_size: number of word tokens on each side of target word
   : param n_files:
   """
   n_pseudodocs = 0
   data = []
   for file_path in crp:
       root = etree.parse(file_path, parser).getroot()
       speeches = speech_iterator(root)
       year = cleanYear(file_path.split('/')[3])
       if speeches != None:
           for speech, mop_id in speeches:
               speech = speech_processor(speech)
               if set(speech).intersection(target):
                   for i, word in enumerate(speech):
                       if word in target:
                           context, pos = create_context(speech, i, window_size)
                           context = ' '.join(context)
                           data.append([n_pseudodocs, context, word, pos, year, mop_id, file_path])
                           n_pseudodocs += 1

   df = pd.DataFrame(data, columns=["doc", "w", "target", "pos", "year", "mop_id", "file_path"])
   return df
 
def main(args):
   with open(args.config, 'r') as f:
       config = yaml.safe_load(f)
   with open(config["target"]) as f:
       targets = json.load(f)
   parser = etree.XMLParser(remove_blank_text=True)
   n_files = count_files(config["corpus"])
 
   for c in config["window_sizes"]:
       start = time()
       for project in config["projects"]:
           crp = corpus_iterator(config["corpus"])
           df = create_contexts(crp, targets["target_"+project], c, n_files, parser)
           df.to_csv(f'{config["data"]}/{project}_window_{c}.csv', index=False)
           print(f'Window size {c} finished for project {project} after {(time()-start)//60} minutes.')
 
if __name__ == '__main__':
   argparser = argparse.ArgumentParser(description=__doc__)
   argparser.add_argument("--config", type=str)
   args = argparser.parse_args()
   main(args)