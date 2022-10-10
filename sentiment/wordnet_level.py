from nltk.corpus import wordnet as wn
import pandas as pd
import numpy as np

def wordnet_level_2D(w1, w2):

    hyper = lambda x: x.hypernyms()

    synsets_w1 = [ syn for syn in wn.synsets(w1) if syn.pos() == 'v']
    synsets_w2 = [ syn for syn in wn.synsets(w2) if syn.pos() == 'v']

    hypers_w1 = [[syn] + list(syn.closure(hyper)) for syn in synsets_w1]
    hypers_w2 = [[syn] + list(syn.closure(hyper)) for syn in synsets_w2]

    result_matrix = []

    for hypers1 in hypers_w1:
        result_line = []
        syns1 = set(hypers1)
        for hypers2 in hypers_w2:
            res = next((i for i in hypers2 if i in syns1), None)
            if res is not None:
                res = (res, hypers1.index(res), hypers2.index(res), hypers1[0], hypers2[0])
            result_line.append(res)
        result_matrix.append(result_line)
    
    return np.array(result_matrix)

def wordnet_level_1D(x):

    if x['classt1'] == 'metaphoric':
        meta_example = x['sentencet1']
        meta_term = x['term1']
        literal_term = x['term2']
    else:
        meta_example = x['sentencet2']
        meta_term = x['term2']
        literal_term = x['term1']
    
    meta_example = meta_example.replace('<b>', '').replace('</b>', '').strip('.').lower()

    hyper = lambda x: x.hypernyms()

    for syn in wn.synsets(meta_term):
        if syn.pos() == 'v':
            syn_examples = [sent.lower().replace("'", '') for sent in syn.examples()]
            if meta_example in syn_examples:
                meta_syn = syn
                break
    else:
        return ['see 2D']

    literal_synsets = [ syn for syn in wn.synsets(literal_term) if syn.pos() == 'v']

    meta_hypers = [syn] + list(meta_syn.closure(hyper))
    literal_hypers = [[syn] + list(syn.closure(hyper)) for syn in literal_synsets]

    result_list = []
    meta_hyper_syns = set(meta_hypers)

    for hypers2 in literal_hypers:
        res = next((i for i in hypers2 if i in meta_hyper_syns), None)
        if res is not None:
            res = (res, meta_hypers.index(res), hypers2.index(res), meta_hypers[0], hypers2[0])
        result_list.append(res)
    
    return result_list

def flatten_result_matrix(result_matrix):
    r = []
    for row in result_matrix:
        for i in row:
            if i is not None:
                r.append(i)
    return r

def flatten_result_list(result_list):
    r = []
    for i in result_list:
        if i is not None:
            r.append(i)
    return r

if __name__ == "__main__":

    df = pd.read_csv('Metaphor-Emotion-Data-Files/Data-Table2-which-is-more-emotional.tsv', sep='\t')
    # find_father = lambda x: flatten_result_matrix(wordnet_level(x['term1'], x['term2']))
    find_father = lambda x: flatten_result_list(wordnet_level_1D(x))

    father = df.apply(find_father, axis=1)
    with open('Metaphor-Emotion-Data-Files/wordnet_level_1D_2.tsv', 'w', encoding='utf-8') as f:
        for line in father:
            line = [ repr(i) for i in line]
            f.write('\t'.join(line)+'\n')
    # df['father'] = father
    # father.to_csv('Metaphor-Emotion-Data-Files/wordnet_level.tsv', sep='\t')