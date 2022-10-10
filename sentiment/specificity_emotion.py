import pandas as pd
from nltk.corpus import wordnet as wn

def obtain_synset(x, ml = 'meta'):
    hyper = x['common_hyper']
    _, _, _, first_synset, second_synset = [ i.strip() for i in hyper.split(',')]
    second_synset = second_synset[:-1]

    if not hyper.startswith('-'):
        meta_synset = first_synset
        literal_synset = second_synset
    else:
        meta_synset = second_synset
        literal_synset = first_synset
    
    if ml == 'meta':
        return meta_synset
    elif ml == 'literal':
        return literal_synset

def find_sister_terms(synset, depth = 0):
    hyper = synset.hypernyms()
    if len(hyper) == 0:
        return []
    if depth == 0:
        hypos = hyper[0].hyponyms()
        hypos.remove(synset)
        return hypos
    else:
        hyper_sisters = find_sister_terms(hyper[0], depth-1)
        hypos = []
        for hyper_sister in hyper_sisters:
            hypos.extend(hyper_sister.hyponyms())
        return hypos

def obtain_sentence(x, ml = 'meta'):
    if x['classt1'] == 'metaphoric':
        meta_example = x['sentencet1']
        meta_term = x['term1']
        literal_term = x['term2']
        literal_example = x['sentencet2']
    else:
        meta_example = x['sentencet2']
        meta_term = x['term2']
        literal_term = x['term1']
        literal_example = x['sentencet1']

    if ml == 'meta':
        return meta_example
    elif ml == 'literal':
        return literal_example

def find_more_specific(synset, depth = 0):
    if depth == 0:
        return synset.hyponyms()
    else:
        sisters = find_sister_terms(synset, depth=depth-1)
        hypos = []
        for sister in sisters:
            hypos.extend(sister.hyponyms())
        return hypos

def to_lemmas(x):
    lemmas = []
    for synset in x:
        lemmas.append([lemma.name() for lemma in synset.lemmas()])
    return lemmas

def produce_similar_specific(df):
    meta_synsets = df.apply(obtain_synset, axis=1)
    synset_object = lambda x: wn.synset(x[8:-2])

    meta_synsets = meta_synsets.apply(synset_object)
    similar_specific = meta_synsets.apply(find_sister_terms)
    similar_specific_terms = similar_specific.apply(to_lemmas)

    meta_sentence = df.apply(obtain_sentence, axis=1)

    with open('Metaphor-Emotion-Data-Files/similar_specific_terms.tsv', 'w') as f:
        for synset, sent, terms in zip(meta_synsets, meta_sentence, similar_specific_terms):
            f.write(str(synset) + '\t' + sent + '\t')
            f.write('\t'.join([', '.join(lemmas) for lemmas in terms]) + '\n')

def produce_more_specific(df):
    synsets = df.apply(obtain_synset, axis=1, ml = 'literal')
    sentences = df.apply(obtain_sentence, axis=1, ml = 'literal')

    synset_object = lambda x: wn.synset(x[8:-2])

    synsets = synsets.apply(synset_object)
    more_specific = synsets.apply(find_more_specific)
    more_specific_terms = more_specific.apply(to_lemmas)

    with open('Metaphor-Emotion-Data-Files/more_specific_terms.tsv', 'w') as f:
        for synset, sent, terms in zip(synsets, sentences, more_specific_terms):
            f.write(str(synset) + '\t' + sent + '\t')
            f.write('\t'.join([', '.join(lemmas) for lemmas in terms]) + '\n')

if __name__ == '__main__':

    df = pd.read_csv('Metaphor-Emotion-Data-Files/emotional_wordnet_1D.tsv', sep='\t')
    df = df.dropna(subset=['specificity'])

    # produce_more_specific(df)
    produce_similar_specific(df)

    # print(similar_specific_terms)