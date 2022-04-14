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
from typing import Union, List
import xml.etree.cElementTree as ET

import sys

@dataclass
class sense:
    lemma: str
    gloss: str
    # synset_name: str
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
    tokens: List[Token]
    index : int
    gloss: str
    # synset_name: str
    examples: List[str] = None

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
            return self.moh_word2lemmas[word]
        senses = []
        for s in wn.synsets(word):
            lemmas = s.lemmas()
            lemma = [l for l in lemmas if l.name() == word][0]
            senses.append(sense(lemma=lemma.key(), gloss=s.definition()))
        assert len(senses), f'No sense found for {word}!'
        return senses
                

class lemma2sentences:
    def __init__(self, source, save_path = 'embeddings/index'):
        # self.word2lemmas = word2lemmas()
        self.source = source
        self._prepare(source, save_path = 'embeddings/index')
    
    def _prepare(self, source, save_path):
        if source == 'semcor':
            self.load_lemma2context_semcor(source, save_path)
        elif source == 'senseval3':
            self.load_lemma2context_senseval3(source, save_path)
        elif source == 'wordnet':
            self.nlp = spacy.load('en_core_web_sm', disable=["parser", "ner"])
    
    def load_lemma2context_senseval3(self, source, save_path):
        lemma2context_pkl = os.path.join(save_path, 'lemma2context_senseval3.pkl')
        dictionary_pkl = os.path.join(save_path, 'dictionary_senseval3.pkl')

        if not os.path.exists(lemma2context_pkl):
            senseval3_data_path = 'wsd/senseval/senseval3/EnglishLS.train/EnglishLS.train'
            sense_dict_path = 'wsd/senseval/senseval3/EnglishLS.train/EnglishLS.dictionary.mapping.xml.new'
            tree = ET.parse(sense_dict_path)
            root = tree.getroot()

            # Read dictionary.mapping.xml file
            dictionary = {}
            for lex in root.iter('lexelt'):
                key = lex.attrib['item']
                dictionary[key] = {}
                for sense in lex.iter('sense'):
                    attribs = sense.attrib
                    id_ = attribs.pop('id')
                    dictionary[key][id_] = attribs
            
            tree = ET.parse(senseval3_data_path)
            root = tree.getroot()
            # Read train.xml file
            lemma2context = {}
            for lex in root.iter('lexelt'):
                key = lex.attrib['item']
                lemma2context[key] = []
                for ins in lex.iter('instance'):
                    answers = [ans.attrib['senseid'] for ans in ins.findall('answer')]
                    if 'U' in answers:
                        answers.remove('U')
                        if not len(answers):
                            print(f'-----no sense id provides for {key}.')
                            continue
                    tokens = []
                    index = -1
                    for idx, word in enumerate(ins.find('context').text.split()):
                        if not (word.startswith('--') and word.endswith('--')):
                            token = Token(word = word, lemma = '_', pos = '_', sense = '_') 
                        else:
                            token = Token(word = word, lemma = '_', pos = '_', sense = ';'.join(answers))
                            index = idx
                        tokens.append(token)
                    assert index !=-1
                    gloss = dictionary[key][answers[0]]['gloss']
                    gloss = [ i for i in gloss[1:-1].split(';') if i ]
                    if len(gloss)>1:
                        examples = gloss[1:]
                    else:
                        examples = None
                    gloss = gloss[0]
                    cont = Context(tokens = tokens, index = index, gloss = gloss, examples=examples)
                    lemma2context[key].append(cont)

            with open(dictionary_pkl, 'wb') as f:
                pickle.dump(dictionary, f)
            with open(lemma2context_pkl, 'wb') as f:
                pickle.dump(lemma2context, f)
            
            self.lemma2context = lemma2context
            self.dictionary = dictionary 

            return

        with open(lemma2context_pkl, 'rb') as f:
            lemma2context = pickle.load(f)
        with open(dictionary_pkl, 'rb') as f:
            dictionary = pickle.load(f)

        self.lemma2context = lemma2context
        self.dictionary = dictionary     

    def load_lemma2context_semcor(self, source, save_path):
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
        
            self.sentences = sentences
            self.lemma2context = lemma2context
            
            return
        
        with open(sentences_pkl, 'rb') as f:
            sentences = pickle.load(f)
        with open(dict_pkl, 'rb') as f:
            lemma2context = pickle.load(f)
        
        self.sentences = sentences
        self.lemma2context = lemma2context
    
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
    
    def get_senseval3_results(self, sense):
        """
        Args:
            sense (str): just word with a pos tag, e.g., activate.v 
            Do not need to be a lemma str
        """
        if sense not in self.lemma2context:
            print(f'Sense {sense} not in the corpus. All avaliable words in Senseval3 are {self.lemma2context.keys()}.')
            return ''
        return self.lemma2context[sense]
    
    def get_semcor_results(self, sense):
        if sense not in self.lemma2context:
            print(f'Sense {sense} not in the corpus.')
            return ''
        return [Context(tokens = self.sentences[i[0]], index = i[1], \
            gloss=wn.lemma_from_key(self.sentences[i[0]][i[1]].sense).synset().definition()) \
            for i in self.lemma2context[sense]]

    def __call__(self, sense):
        if self.source == 'wordnet':
            return self.get_wn_examples(sense)
        elif self.source == 'senseval3':
            return self.get_senseval3_results(sense)
        elif self.source == 'semcor':
            return self.get_semcor_results(sense)

class word2sentence:
    valid_sources = ['semcor', 'senseval3', 'wordnet']

    def __init__(self, source, index_path = 'embeddings/index'):
        self._check_source(source)
        self.source = source
        self._prepare_indexs(source, index_path)
    
    def _prepare_indexs(self, source, index_path):
        if source == 'semcor':
            self.word2lemmas = word2lemmas(save_path=index_path)
            self.lemma2context = lemma2sentences(source = source, save_path=index_path)
        elif source == 'senseval3':
            self.lemma2context = lemma2sentences(source = source, save_path=index_path)

    def _check_source(self, source):
        assert source in self.valid_sources, f'{source} is not a valid source, please use {self.valid_sources}'
    
    def remove_rare_context(self, contexts, minimum):
        if not minimum:
            return contexts
        gloss2context = {}
        for cont in contexts:
            gloss = cont.gloss
            if gloss not in gloss2context: gloss2context[gloss] = []
            gloss2context[gloss].append(cont)
        pruned_contexts = []
        for gloss, conts in gloss2context.items():
            if len(conts)>minimum:
                pruned_contexts.extend(conts)
        return pruned_contexts

    def __call__(self, word, minimum = 0):
        if self.source == 'senseval3':
            sentences = self.lemma2context(word)
        else:
            lemmas = self.word2lemmas(word)
            # sentences = {lemma.lemma: {'class': lemma.label, 'sentences': self.lemma2context(lemma.lemma, source=self.source), 'gloss': wn.lemma_from_key(lemma.lemma).synset().definition()} for lemma in lemmas}
            sentences = []
            for lemma in lemmas:
                sentences.extend(self.lemma2context(lemma.lemma))
        return self.remove_rare_context(sentences, minimum)        

class synset2sentence:

    def __init__(self, tokenizer, index_path = 'embeddings/index'):
        self._load_synset2sentence_map(index_path)
        self._load_sentences(index_path)
        self.tokenizer = tokenizer
        self._encoding_sentences(index_path)
    
    def _encoding_sentences(self, index_path):
        sentences_encoding_pkl = os.path.join(index_path, 'sentences_encoding.pkl')
        if not os.path.exists(sentences_encoding_pkl):
            sentences_encoding = []
            for sent in self.sentences:
                tokens = [t.word for t in sent]
                encoding = self.tokenizer(tokens, is_split_into_words = True)
                idxs = encoding.word_ids()
                sentences_encoding.append({'encoding': encoding, 'word_ids': idxs})
            self.sentences_encoding = sentences_encoding

            with open(sentences_encoding_pkl, 'wb') as f:
                pickle.dump(sentences_encoding, f)
        
            return
        
        with open(sentences_encoding_pkl, 'rb') as f:
            self.sentences_encoding = pickle.load(f)
    
    def _load_sentences(self, index_path):
        sentences_pkl = f'{index_path}/sentences.pkl'
        assert os.path.exists(sentences_pkl)
        with open(sentences_pkl, 'rb') as f:
            self.sentences = pickle.load(f)
    
    def _load_synset2sentence_map(self, index_path):
        synset2sentence_pkl = os.path.join(index_path, 'synset2sentence.pkl')
        if not os.path.exists(synset2sentence_pkl):
            assert os.path.exists(f'{index_path}/lemma2context.pkl'), 'lemma2context.pkl mapping does not exist'
            with open(f'{index_path}/lemma2context.pkl', 'rb') as f:
                lemma2context = pickle.load(f)
            
            synset2sentence = {}
            for lemma, sents in lemma2context.items():
                synset_name = wn.lemma_from_key(lemma).synset().name()
                if synset_name not in synset2sentence: synset2sentence[synset_name] = []
                synset2sentence[synset_name].extend(sents)
            
            with open(synset2sentence_pkl, 'wb') as f:
                pickle.dump(synset2sentence, f)
            
            self.synset2sentence = synset2sentence

            return
        
        with open(synset2sentence_pkl, 'rb') as f:
            self.synset2sentence = pickle.load(f)
    
    def get_all_synsets(self):
        return self.synset2sentence.keys()
    
    def synsets_frequencies(self):
        return {synset: len(sents) for synset, sents in self.synset2sentence.items()}
    
    def __call__(self, synset_name, max_length):
        assert synset_name in self.synset2sentence
        results = []
        for sent in self.synset2sentence[synset_name]:
            sentence_id = sent[0]
            index = sent[1]
            if len(self.sentences_encoding[sentence_id]['word_ids'])>max_length:
                continue
            new_idx = self.sentences_encoding[sentence_id]['word_ids'].index(index)
            results.append({'encoding': self.sentences_encoding[sentence_id]['encoding'], 'idx': new_idx})
        return results
    
    def realized_to_context(self, sentences):
        return [Context(tokens = self.sentences[i[0]], index = i[1], \
            gloss=wn.lemma_from_key(self.sentences[i[0]][i[1]].sense).synset().definition()) \
            for i in sentences]

if __name__ == '__main__':
    word2sentence = word2sentence()
    contexts = []
    for k,v in word2sentence('act').items():
        contexts.extend(v['sentences'])

    demo = SemanticEmbedding('roberta-base', kernel_bias_path='embedding/kernel', dynamic_kernel=True)
    # print(demo.get_embeddings(contexts), [str(con.tokens[con.index].sense) for con in contexts])
    # plotPCA(demo.get_embeddings(contexts), [str(con.tokens[con.index].sense) for con in contexts])