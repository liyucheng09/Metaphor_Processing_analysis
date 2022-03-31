import pickle
import os
import pandas as pd
from nltk.corpus import wordnet as wn
from dataclasses import dataclass
from pprint import pprint
import spacy
import torch
# from lyc.visualize import plotPCA
from lyc.utils import get_model, get_tokenizer, vector_l2_normlize

import sys

@dataclass
class sense:
    lemma: str
    gloss: str
    label: str = None
    confidence: float = None

@dataclass
class Token:
    word: str
    lemma: str
    pos: str
    sense: str

@dataclass
class Context:
    tokens: list[Token]
    index : int
    gloss: str

    def __repr__(self):
        return ' '.join([t.word for t in self.tokens])

def get_lemma(x):
    """
    x: MOH sense string
    """
    try:
        lemma, pos, sense_id = x.split('#')
    except:
        return None
    sense_id = int(sense_id)
    lemma_str = f'{lemma}.{pos}.{sense_id:02d}.{lemma}'
    return wn.lemma(lemma_str).key()

class word2lemmas:
    def __init__(self, save_path = 'embeddings/index'):
        self.moh_word2lemmas = self.load_moh_word2lemma_dict(save_path)
    
    def load_moh_word2lemma_dict(self, save_path):
        pkl_path = os.path.join(save_path, 'moh_word2lemmas.pkl')
        if not os.path.exists(pkl_path):
            word2lemmas = {}
            print(f'No dict found in {pkl_path}, start generating now...')
            moh_path = 'Metaphor-Emotion-Data-Files/Data-metaphoric-or-literal.txt'
            df = pd.read_csv(moh_path, sep='\t')
            for word, sub_df in df.groupby('term'):
                assert word not in word2lemmas
                senses = []
                for _, i in sub_df.iterrows():
                    lemma = get_lemma(i['sense'])
                    senses.append(sense(lemma = lemma, gloss=wn.lemma_from_key(lemma).synset().definition(), label = i['class'], confidence = i['confidence']))
                word2lemmas[word] = senses
            
            with open(pkl_path, 'wb') as f:
                pickle.dump(word2lemmas, f)
            print(f'saved to {pkl_path}')
            return word2lemmas
        with open(pkl_path, 'rb') as f:
            word2lemmas = pickle.load(f)
        return word2lemmas

    def __call__(self, word):
        if word in self.moh_word2lemmas:
            return self.word2lemmas[word]
        senses = []
        for s in wn.synsets(word):
            lemmas = s.lemmas()
            lemma = [l for l in lemmas if l.name() == word][0]
            senses.append(sense(lemma=lemma.key(), gloss=s.definition()))
        assert len(senses), f'No sense found for {word}!'
        return senses
                

class lemma2sentences:
    def __init__(self, save_path = 'embeddings/index'):
        self.word2lemmas = word2lemmas()
        self.sentences, self.lemma2context = self.load_lemma_to_context(save_path = 'embeddings/index')
        self.nlp = spacy.load('en_core_web_sm', disable=["parser", "ner"])
    
    def load_lemma_to_context(self, save_path):
        sentences_pkl, dict_pkl = os.path.join(save_path, 'sentences.pkl'), os.path.join(save_path, 'lemma2context.pkl')
        if not os.path.exists(sentences_pkl):
            semcor_path = '/Users/yucheng/projects/wsd-biencoders/WSD_Evaluation_Framework/conll/semcor.conll'
            sentences = []
            lemma2context = {}
            with open(semcor_path, 'r', encoding='utf-8') as f:
                for sent in f.read().split('\n\n'):
                    if not sent: continue
                    tokens = []
                    for token in sent.split('\n'):
                        if not token: continue
                        w, l, pos, id_, sense = token.split('\t')
                        tokens.append(Token(word = w, lemma = l, pos = pos, sense = sense))
                    sentences.append(tokens)
                    for index, token in enumerate(tokens):
                        if token.sense != '-1':
                            if token.sense not in lemma2context: lemma2context[token.sense] = []
                            lemma2context[token.sense].append((len(sentences) - 1, index))
            
            with open(sentences_pkl, 'wb') as f:
                pickle.dump(sentences, f)
            with open(dict_pkl, 'wb') as f:
                pickle.dump(lemma2context, f)
            print(f'saved to {sentences_pkl}, {dict_pkl}.')
        
            return sentences, lemma2context
        
        with open(sentences_pkl, 'rb') as f:
            sentences = pickle.load(f)
        with open(dict_pkl, 'rb') as f:
            lemma2context = pickle.load(f)
        
        return sentences, lemma2context
    
    def get_wn_examples(self, sense):
        meaning = wn.lemma_from_key(sense).synset()
        examples = meaning.examples()
        if len(examples) == 0:
            print(f'Sense {sense} has no examples in wordnet.')
            return ''
        sentences = []
        all_lemmas = meaning.lemmas()
        for sent in examples:
            sent = [Token(word = t.text, lemma=t.lemma_, pos=t.pos_, sense='-1') for t in self.nlp(sent)]
            for lemma in all_lemmas:
                for idx, t in enumerate(sent):
                    if lemma.name() == t.lemma:
                        t.sense = lemma
                        sent[idx] = t
                        sentences.append(Context(tokens = sent, index=idx))
                        break
                else:
                    continue
                break
        return sentences

    def __call__(self, sense, method = 'wsd'):
        if method == 'wordnet':
            return self.get_wn_examples(sense)
        if sense not in self.lemma2context:
            print(f'Sense {sense} not in the corpus.')
            return ''
        return [Context(tokens = self.sentences[i[0]], index = i[1], \
            gloss=wn.lemma_from_key(self.sentences[i[0]][i[1]].sense).synset().definition()) \
            for i in self.lemma2context[sense]]

class word2sentence:
    def __init__(self, save_path = 'embeddings/index'):
        self.word2lemmas = word2lemmas(save_path=save_path)
        self.lemma2context = lemma2sentences(save_path=save_path)
    
    def __call__(self, word):
        lemmas = self.word2lemmas(word)
        sentences = {lemma.lemma: {'class': lemma.label, 'sentences': self.lemma2context(lemma.lemma, method='wsd'), 'gloss': wn.lemma_from_key(lemma.lemma).synset().definition()} for lemma in lemmas}
        return sentences


if __name__ == '__main__':
    word2sentence = word2sentence()
    contexts = []
    for k,v in word2sentence('act').items():
        contexts.extend(v['sentences'])

    demo = SemanticEmbedding('roberta-base', kernel_bias_path='embedding/kernel', dynamic_kernel=True)
    # print(demo.get_embeddings(contexts), [str(con.tokens[con.index].sense) for con in contexts])
    # plotPCA(demo.get_embeddings(contexts), [str(con.tokens[con.index].sense) for con in contexts])