import pickle
import os
import pandas as pd
from nltk.corpus import wordnet as wn
from dataclasses import dataclass
from utils import get_lemma
from pprint import pprint
import spacy
import torch

import sys
sys.path.append('/Users/yucheng/projects/sentence_embedding/')
from demo import SentenceEmbedding

@dataclass
class sense:
    lemma: str
    label: str
    confidence: float

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

    def __repr__(self):
        return ' '.join([t.word for t in self.tokens])

class word2lemmas:
    def __init__(self, save_path = 'embeddings/index'):
        self.word2lemmas = self.load_word2lemma_dict(save_path)
    
    def load_word2lemma_dict(self, save_path):
        pkl_path = os.path.join(save_path, 'word2lemmas.pkl')
        if not os.path.exists(pkl_path):
            word2lemmas = {}
            print(f'No dict found in {pkl_path}, start generating now...')
            moh_path = 'Metaphor-Emotion-Data-Files/Data-metaphoric-or-literal.txt'
            df = pd.read_csv(moh_path, sep='\t')
            for word, sub_df in df.groupby('term'):
                assert word not in word2lemmas
                senses = [sense(lemma = get_lemma(i['sense']), label = i['class'], confidence = i['confidence']) for _, i in sub_df.iterrows()]
                word2lemmas[word] = senses
            
            with open(pkl_path, 'wb') as f:
                pickle.dump(word2lemmas, f)
            print(f'saved to {pkl_path}')
            return word2lemmas
        with open(pkl_path, 'rb') as f:
            word2lemmas = pickle.load(f)
        return word2lemmas

    def __call__(self, word):
        return self.word2lemmas[word]
                

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
        return [Context(tokens = self.sentences[i[0]], index = i[1]) for i in self.lemma2context[sense]]

class word2sentence:
    def __init__(self, save_path = 'embeddings/index'):
        self.word2lemmas = word2lemmas(save_path=save_path)
        self.lemma2context = lemma2sentences(save_path=save_path)
    
    def __call__(self, word):
        lemmas = self.word2lemmas(word)
        sentences = {lemma.lemma: {'class': lemma.label, 'sentences': self.lemma2context(lemma.lemma, method='wordnet'), 'gloss': wn.lemma_from_key(lemma.lemma).synset().definition()} for lemma in lemmas}
        return sentences

class SemanticEmbedding(SentenceEmbedding):
    def __init__(self, model_path, max_length=64, n_components=768, kernel_bias_path=None, 
        corpus_for_kernel_computing=None, pool='last_idx', dynamic_kernel = False):
        """[summary]

        Args:
            model_path ([type]): [description]
            max_length (int, optional): [description]. Defaults to 64.
            n_components (int, optional): [description]. Defaults to 768.
            kernel_bias_path ([type], optional): [description]. Defaults to None.
            corpus_for_kernel_computing ([type], optional): 训练kernel和bias需要的语料，纯txt，一行一个句子. Defaults to None.
        """
        self.model = get_model(Sentence, model_path)
        self.tokenizer = get_tokenizer(model_path)
        self.max_length=max_length
        self.n_components=n_components
        self.pool=pool
        self.dynamic_kernel = dynamic_kernel
        if not dynamic_kernel:
            self._get_kernel_and_bias(model_path, kernel_bias_path, corpus_for_kernel_computing)

    def preprocess_context(self, contexts: list(Context)):
        sents = []
        idxs = []
        senses = []
        new_idxs = []

        for cont in contexts:
            idx = cont.index
            target_sense = cont[idx].sense
            sent = [t.word for t in cont]
            
            sents.append(sent)
            idxs.append(idx)
            senses.append(target_sense)
        
        output = self.tokenizer(sents, is_split_into_words = True, max_length = self.max_length, padding=True, truncation = True, return_tensor = 'pt')
        for i in range(len(sent)):
            new_idx = output.word_ids(i)
            new_idxs.append(new_idx)
        
        new_idxs = torch.LongTensor(new_idxs)
        
        return output, new_idxs, senses

    def get_embeddings(self, contexts: list(Context), whitening=False):
        encoding, idxs, senses = self.preprocess_context(contexts)
        with torch.no_grad():
            vecs=get_vectors(self.model, tokenized_sents, pool=self.pool, idxs = idxs)[0]
        vecs=vecs.cpu().numpy()

        if whitening:
            if self.dynamic_kernel:
                vecs=vector_l2_normlize(vecs)
                kernel, bias = compute_kernel_bias([vecs])
            else:
                kernel, bias = self.kernel, self.bias
            kernel=kernel[:, :self.n_components]
            vecs=transform_and_normalize(vecs, kernel, bias)
            return vecs
        vecs=vector_l2_normlize(vecs)
        return vecs
    
    def _computing_kernel_and_save(self, kernel_bias_path, corpus_for_kernel_computing):
        with open(corpus_for_kernel_computing, 'rb') as f:
            contexts = pickle.load(f)
        vecs=self.get_embeddings(contexts, whitening=False)
        kernel, bias = compute_kernel_bias([vecs])
        save_kernel_and_bias(kernel, bias, kernel_bias_path)

if __name__ == '__main__':
    word2sentence = word2sentence()
    contexts = []
    for k,v in word2sentence('act').items():
        contexts.extend(v['sentences'])

    demo = SemanticEmbedding('roberta-base', kernel_bias_path='embedding/kernel', dynamic_kernel=True)
    demo.get_embeddings(contexts)