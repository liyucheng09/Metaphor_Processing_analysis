import sys
sys.path.append('/user/HS502/yl02706/mpa')
from main import SenseEmbedding
from lyc.utils import get_tokenizer, get_model
from util import word2lemmas, sense, Token, Context, word2sentence
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
import os
from lyc.visualize import plotDimensionReduction

def merging(vecs, eps = 0.1, min_samples = 4):
    sims = 1.0001 - np.matmul(vecs, vecs.T)
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(sims).labels_

    return labels

if __name__ == '__main__':
    cwd, max_length, model_path, pool, source, eps, min_samples, = sys.argv[1:]
    max_length = int(max_length)
    eps = float(eps)
    min_samples = int(min_samples)

    index_path = os.path.join(cwd, 'embeddings/index')

    tokenizer = get_tokenizer('roberta-base', add_prefix_space=True)
    model = SenseEmbedding(model_path, add_prefix_space = True, pool = pool, max_length=max_length, output_hidden_states = True)
    word2sentence = word2sentence(source, tokenizer, index_path=index_path)

    if source == 'semcor':
        words = word2sentence.word2lemmas.moh_word2lemmas
        lemma2gloss = lambda lemma: wn.lemma_from_key(lemma).synset().definition()
    elif source == 'senseval3':
        words = word2sentence.word2lemmas.senseval3_word2lemmas
        glosses = {s.lemma : s.gloss for word, senses in words.items() for s in senses }
        lemma2gloss = glosses.get
        sense_merging_mapping = word2sentence.word2lemmas.senseval3_sense_merging
    
    model_id = os.path.basename(model_path)
    for word in words:
        contexts = word2sentence(word, minimum = 1, max_length = max_length)
        if not contexts:
            print(f'{word} do not have enough contexts!')
            continue
        lemmas = [ cont.tokens[cont.index].sense for cont in contexts]
        vecs = model.get_embeddings(contexts)
        labels = merging(vecs, eps=eps, min_samples = min_samples)

        merged_img_path = os.path.join(cwd, 'embeddings/imgs/merging', f'merged_{source}_{eps}_{min_samples}_{word}.png')
        plotDimensionReduction(vecs, labels, merged_img_path)
        print(f'{word} process finished!')