import numpy as np
import pandas as pd
import os, json
from pygibbs import utils
import matplotlib.pyplot as plt
import time
from collections import Counter
import gzip
import sklearn.preprocessing
import progressbar

def get_senses():
    z = np.zeros(M, dtype=int)
    for m in progressbar.progressbar(range(M)):
        z[m] = state.loc[state["#doc"] == m].iloc[data.loc[m,"pos"]]["topic"]
    return z

def rare_words():
    """
    TODO: Add args and move
    """
    thresh = 5
    word_counts = Counter(' '.join(data["w"]).split())
    for m, row in data.iterrows():
        words = []
        for i, word in enumerate(row["w"].split()):
            if word_counts.get(word) > thresh or word in target_words:
                words.append(word)
            elif i < row["pos"]:
                data.loc[m, "pos"] -= 1
        data.loc[m,"w"] = ' '.join(words)
        if len(words) == 0:
            print(f'Document {i} is empty after filtering rare words and is removed.')
            data = data.drop(i)
    data.reset_index(drop=True)
    data["doc"] = data.index

    data.to_csv(os.path.join(path_model, 'data.csv'), index=False)

def extract_params(state, V):
    """
    Extracts model parameters and count matrices from state df.
    """
    M, K = max(state["#doc"])+1, len(set(state["topic"]))
    Nd = np.zeros((M, K))
    Nk = np.zeros((K, V))
    np.add.at(Nd, tuple((state["#doc"], state["topic"])), 1)
    np.add.at(Nk, tuple((state["topic"], state["typeindex"])), 1)
    theta = Nd / Nd.sum(axis=1)[:,np.newaxis]
    phi = Nk / Nk.sum(axis=1)[:,np.newaxis]
    return (theta, phi, Nd, Nk)

def add_meta_data_to_state(state, meta):
    variables = ["year", "gender", "party_abbrev", "lemmas"]
    doclenghts = Counter(state["#doc"])
    data = []
    for m in progressbar.progressbar(range(max(state["#doc"])+1)):
        data.extend([meta.loc[m, variables].tolist()] * doclenghts[m])
    data = pd.DataFrame(data, columns=variables)
    state[variables] = data

with open('data/stopwords-sv.txt','r') as f:
    stopwords = f.read().splitlines()
with open('../riksdagen-corpus/corpus/party_mapping.json', 'r') as f:
    party_map = json.load(f)
periods = np.array([list(np.arange(1920+(10*i), 1920+(10*(i+1)))) for i in range(11)])
colors = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1',\
          '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']
alpha, beta = 0.5, 0.5

def main():
    for p in ['f', 'j']:
        for c in [20, 10, 5]:
            for k in [50, 20, 10, 5]:
                if c == 5 and (k==20 or k==50):
                    continue
                start = time.time()
                # Load data
                with open('data/target-words.json', 'r') as f:
                    target_words = json.load(f)["target_" + p]

                path_project = f'results/lda-models/{p}'
                path_model = f'results/lda-models/{p}/window_{c}_topic_{k}/model'
                path_analysis = f'results/lda-models/{p}/window_{c}_topic_{k}/analysis'
                path_meta = f'results/lda-models/{p}/meta.csv'
                path_state = os.path.join(path_model, 'state.gz')

                color = colors if k <= 10 else None

                # Load global metadata
                meta = pd.read_csv(path_meta)
                meta["z"] = np.load(os.path.join(path_model, 'z.npy'))
                
                # Add global metadata to local dataset
                data = pd.read_csv(f'{path_model}/data.csv')
                data[['lemmas', 'gender', 'party_abbrev', 'z']] =\
                meta[['lemmas', 'gender', 'party_abbrev', 'z']]
                meta = data

                # Load state and add metadata
                state = pd.read_csv(path_state, compression='gzip', sep=' ', skiprows=[1,2])
                add_meta_data_to_state(state, meta)
                state["z"] = state["topic"]

                # Load model
                lp = np.load(f'{path_model}/logperplexity.npy')
                vocab = dict(zip(state.type, state.typeindex))
                M, V, K = len(meta), len(vocab), len(set(state["topic"]))
                theta, phi, Nd, Nk = extract_params(state, V)

                # Sort z
                meta["z"], theta, phi, Nd, Nk, z_sorted = utils.z_sorter(meta["z"], theta, phi, Nd, Nk)
                state = state.replace({'z':{key:value for key, value in zip(list(range(K)), z_sorted)}})

                lemmas = set(meta["lemmas"])
                targets = list(set(meta["target"]))
                genders = ["woman", "man"]
                parties = [p for p in list(set(meta["party_abbrev"])) if str(p) != 'nan']
                labels = {"targets":meta, "corpus":state}

                # Extremely global
                #f = utils.word_freq_over_time(meta["year"], meta["lemmas"], color)
                #plt.savefig(f'{path_project}/lemmas_over_time.png', dpi=300, bbox_inches='tight')
                #plt.close('all')
                #
                ## Global
                out = path_analysis
                exec('try:os.mkdir(out)\nexcept:pass')
                #f = utils.convergence_plot(lp, color, xticks=np.arange(0,10000,10))
                #plt.savefig(f'{out}/logdensity.png', dpi=300, bbox_inches='tight')
                #plt.close('all')
                #
                #distinctwords = utils.distinctWords(Nk, vocab, K, n=30, beta=beta)
                #distinctwords.to_csv(f'{out}/distinct_words.csv', index=False)
                #
                #topwords = utils.topWords(phi, vocab, K, stopwords)
                #topwords.to_csv(f'{out}/top_words.csv', index=False)
                #
                #topwords = utils.topWords(phi, vocab, K, stopwords, targets)
                #topwords.to_csv(f'{out}/top_words_targets_removed.csv', index=False)
                #out = path_analysis
                #topdocs = utils.lda_top_docs(meta, theta, n=20)
                #with open(f'{out}/top_docs_by_topic.txt', 'w') as f:
                # for line in topdocs:
                #     f.write(line + '\n'*2)

                # Dataset
                for label, dataset in labels.items():               
                    out = os.path.join(path_analysis, label)
                    exec('try:os.mkdir(out)\nexcept:pass')
#                    f = utils.senses_over_time(dataset["year"], dataset["z"], color)
#                    plt.savefig(f'{out}/senses_over_time.png', dpi=300, bbox_inches='tight')
#                    plt.close('all')
                    topicprops = utils.topic_props_over_time_table(time=dataset["year"], z=dataset["z"], periods=periods)
                    topicprops.to_csv(f'{out}/topic_props_over_time_rel.csv')
                    topicprops = utils.topic_props_over_time_table(time=dataset["year"], z=dataset["z"], periods=periods, relative=False)
                    topicprops.to_csv(f'{out}/topic_props_over_time_abs.csv')


                    # Dataset, gender
                    out = os.path.join(path_analysis, label, "gender")
                    exec('try:os.mkdir(out)\nexcept:pass')
                    for gender in genders:

#                        f = utils.senses_over_time(dataset["year"], dataset["z"], color, meta=dataset, variable="gender", value=gender)
#                        f.suptitle('Women' if gender == "woman" else "Men")
#                        gender_abbrev = "m" if gender == "man" else "f"
#                        plt.savefig(f'{out}/senses_over_time_{gender_abbrev}.png', dpi=300, bbox_inches='tight')
#                        plt.close('all')
#
                        topicprops = utils.topic_props_over_time_table(\
                            time=dataset["year"], z=dataset["z"], periods=periods,\
                            meta=dataset, variable='gender', value=gender)
                        topicprops.to_csv(f'{out}/topic_props_over_time_{gender}_rel.csv')
                        topicprops = utils.topic_props_over_time_table(\
                            time=dataset["year"], z=dataset["z"], periods=periods,\
                            meta=dataset, variable='gender', value=gender, relative=False)
                        topicprops.to_csv(f'{out}/topic_props_over_time_{gender}_abs.csv')

                    # Dataset, party
                    out = os.path.join(path_analysis, label, "party")
                    exec('try:os.mkdir(out)\nexcept:pass')
                    for party in parties:
#                        f = utils.senses_over_time(dataset["year"], dataset["z"], color, dataset, "party_abbrev", party, xlim=False)
#                        f.suptitle(max([key for key, value in party_map.items() if value == party]))
#                        plt.savefig(f'{out}/senses_over_time_{party}.png', dpi=300, bbox_inches='tight')
#                        plt.close('all')

                        topicprops = utils.topic_props_over_time_table(\
                            time=dataset["year"], z=dataset["z"], periods=periods,\
                            meta=dataset, variable='party_abbrev', value=party)
                        topicprops.to_csv(f'{out}/topic_props_over_time_{party}_rel.csv')
                        topicprops = utils.topic_props_over_time_table(\
                            time=dataset["year"], z=dataset["z"], periods=periods,\
                            meta=dataset, variable='party_abbrev', value=party, relative=False)
                        topicprops.to_csv(f'{out}/topic_props_over_time_{party}_abs.csv')

                    # Dataset, lemma
                    if label == "targets":
                        out = os.path.join(path_analysis, label, "lemmas")
                        exec('try:os.mkdir(out)\nexcept:pass')
                        lemmaprops = utils.lemma_props_by_topic(dataset["lemmas"], dataset["z"])
                        lemmaprops.to_csv(f'{out}/lemma_topic_distributions_rel.csv')
                        lemmaprops = utils.lemma_props_by_topic(dataset["lemmas"], dataset["z"], relative=False)
                        lemmaprops.to_csv(f'{out}/lemma_topic_distributions_abs.csv')

                        for lemma in lemmas:
                            data = dataset.loc[dataset["lemmas"] == lemma]
                            out = os.path.join(path_analysis, label, "lemmas", lemma)
                            exec('try:os.mkdir(out)\nexcept:pass')
#                            f = utils.senses_over_time(data["year"], data["z"], color)
#                            plt.savefig(f'{out}/senses_over_time.png', dpi=300, bbox_inches='tight')
#                            plt.close('all')

                            topicprops = utils.topic_props_over_time_table(time=data["year"], z=data["z"], periods=periods)
                            topicprops.to_csv(f'{out}/topic_props_over_time_rel.csv')
                            topicprops = utils.topic_props_over_time_table(time=data["year"], z=data["z"], periods=periods, relative=False)
                            topicprops.to_csv(f'{out}/topic_props_over_time_abs.csv')

                            # Dataset, lemma, gender
                            out = os.path.join(path_analysis, label, "lemmas", lemma, "gender")
                            exec('try:os.mkdir(out)\nexcept:pass')
                            for gender in genders:
#                                f = utils.senses_over_time(data["year"], data["z"], color, meta=data, variable="gender", value=gender)
#                                f.suptitle('Women' if gender == "woman" else "Men")
#                                gender_abbrev = "m" if gender == "man" else "f"
#                                plt.savefig(f'{out}/senses_over_time_{gender_abbrev}.png', dpi=300, bbox_inches='tight')
#                                plt.close('all')

                                topicprops = utils.topic_props_over_time_table(\
                                    time=data["year"], z=data["z"], periods=periods,\
                                    meta=data, variable='gender', value=gender)
                                topicprops.to_csv(f'{out}/topic_props_over_time_{gender}_rel.csv')
                                topicprops = utils.topic_props_over_time_table(\
                                    time=data["year"], z=data["z"], periods=periods,\
                                    meta=data, variable='gender', value=gender, relative=False)
                                topicprops.to_csv(f'{out}/topic_props_over_time_{gender}_abs.csv')

                            # Dataset, lemma, party
                            out = os.path.join(path_analysis, label, "lemmas", lemma, "party")
                            exec('try:os.mkdir(out)\nexcept:pass')
                            for party in parties:
#                                f = utils.senses_over_time(data["year"], data["z"], color, data, "party_abbrev", party, xlim=False)
#                                f.suptitle(max([key for key, value in party_map.items() if value == party]))
#                                plt.savefig(f'{out}/senses_over_time_{party}.png', dpi=300, bbox_inches='tight')
#                                plt.close('all')

                                topicprops = utils.topic_props_over_time_table(\
                                    time=data["year"], z=data["z"], periods=periods,\
                                    meta=data, variable='party_abbrev', value=party)
                                topicprops.to_csv(f'{out}/topic_props_over_time_{party}_rel.csv')
                                topicprops = utils.topic_props_over_time_table(\
                                    time=data["year"], z=data["z"], periods=periods,\
                                    meta=data, variable='party_abbrev', value=party, relative=False)
                                topicprops.to_csv(f'{out}/topic_props_over_time_{party}_abs.csv')

                print(f'Project: {p}, window: {c}, topic: {k} finished after {round(time.time()-start)} seconds.')

main()