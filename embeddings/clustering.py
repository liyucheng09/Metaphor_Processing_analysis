import sys
sys.path.append('/user/HS502/yl02706/mpa')
from lyc.model import simcse
from lyc.utils import get_tokenizer, get_model
from lyc.data import SentenceDataset
from lyc.visualize import plotDimensionReduction
import pandas as pd
from util import word2lemmas, sense, Token, Context, word2sentence
from main import SenseEmbedding
import numpy as np
import os
from nltk.corpus import wordnet as wn
from collections import Counter

def compute_overlapping(vecs, lemmas):
    """计算每个sense和其余sense之间的重叠程度。
    假设sense A共有N个contexts，那么我们令sense A的每个点寻找距离起其最近的N个邻居。
    将这N个邻居中sense A的比例，作为sense A与其余sense重合度的一种度量。

    Args:
        vecs (np.array): list of representation
        lemmas (np.array): list of lemma strings

    Returns:
        overlapping (dict): {lemma: overlap(__float__) }
    """
    num_vecs = vecs.shape[0]
    sims = np.matmul(vecs, vecs.T) - np.eye(num_vecs)*100
    sort_idxs = sims.argsort(axis=1)

    overlapping = {}
    for lemma in set(lemmas):
        xidx = (lemmas == lemma)
        idxs_of_lemma = xidx.nonzero()[0]
        num_of_context = xidx.sum()
        lemma_knn = sort_idxs[xidx][:, :num_of_context-1]
        mask = np.isin(lemma_knn, idxs_of_lemma)
        total = num_of_context * (num_of_context-1)

        overlapping[lemma] = mask.sum()/total
    
    return overlapping

def prepare_overlapping_computing_senseval3(vecs, lemmas):
    """This function is intented to deal with words with multiple sense annotated in senseval3.
    The basic idea here is to duplicate the vector for each sense.

    Args:
        vecs (_type_): vectors np.array
        lemmas (_type_): lemmas np.array
    """

    idxs = list(range(len(lemma)))
    extended_lemmas = []
    for index, lemma in enumerate(lemmas):
        if ';' not in lemma:
            continue
        ls = lemma.split(';')
        extended_lemmas.extend(ls)
        idxs.extend([index]*len(ls))
    
    return vecs[idxs], np.append(lemmas, extended_lemmas)


if __name__ == '__main__':
    cwd, max_length, model_path, pool, source = sys.argv[1:]
    max_length = int(max_length)

    index_path = os.path.join(cwd, 'embeddings/index')

    tokenizer = get_tokenizer('roberta-base', add_prefix_space=True)
    model = SenseEmbedding(model_path, add_prefix_space = True, pool = pool, max_length=max_length, output_hidden_states = True)
    word2sentence = word2sentence(source, tokenizer, index_path=index_path)

    if source == 'semcor':
        words = word2sentence.word2lemmas.moh_word2lemmas
    elif source == 'senseval3':
        words = word2sentence.word2lemmas.senseval3_word2lemmas
    lemma2gloss = {s.lemma : s.gloss for word, senses in words.items() for s in senses }

    # model = get_model(simcse, model_path, pool = pool, output_hidden_states = True)
    for word in words:
        contexts = word2sentence(word, minimum = 1, max_length = max_length)
        if not contexts:
            print(f'{word} do not have enough contexts!')
            continue
        lemmas = [ cont.tokens[cont.index].sense for cont in contexts]
        lemma_counter = Counter(lemmas)
        lemmas = np.array(lemmas)
        metaphorical_senses = [ sense.lemma for sense in words[word] if sense.label == 'metaphorical']

        vecs = model.get_embeddings(contexts)
        if source == 'senseval3':
            vecs, lemmas = prepare_overlapping_computing_senseval3(vecs, lemmas)
        overlapping = compute_overlapping(vecs, lemmas)

        model_id = os.path.basename(model_path)
        output_overlapping_path = os.path.join(cwd, 'embeddings/overlapping', f'{model_id}_{word}.result')
        img_path = os.path.join(cwd, 'embeddings/imgs/clustering', f'{model_id}_{word}.png')
        
        overlap_path = open(output_overlapping_path, 'w', encoding='utf-8')
        overlap_path.write(f'lemma\toverlap_score\tgloss\tnum_sent\n')
        for lemma, overlap in overlapping.items():
            gloss = lemma2gloss[lemma]
            if lemma in metaphorical_senses:
                lemma = '*' + lemma
            overlap_path.write(f'{lemma}\t{overlap}\t{gloss}\t{lemma_counter[lemma]}\n')
        overlap_path.close()

        plotDimensionReduction(vecs, lemmas, img_path)
        print(f'{word} process finished!')        