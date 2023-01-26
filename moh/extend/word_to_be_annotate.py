from nltk.corpus import wordnet as wn
import spacy
from embeddings.util import *
import pickle
import os

class lemma_to_annotate:

    def __init__(self, pos, min_senses = 3, max_senses = 10, min_wn_examples = 1, min_semcor_examples_per_sense = 3, min_semcor_examples_per_word = True,
                    save_path = 'moh/extend/'):
        self.nlp = spacy.load('en_core_web_sm', disable=["parser", "ner"])
        with open('embeddings/index/lemma2context.pkl', 'rb') as f:
            self.lemma2context = pickle.load(f)
        self.pos = pos
        self.min_senses = min_senses
        self.max_senses = max_senses
        self.min_wn_examples = min_wn_examples
        self.min_semcor_examples_per_sense = min_semcor_examples_per_sense
        self.min_semcor_examples_per_word = min_semcor_examples_per_word

        self.all_lemmas = []
        self.get_lemmas_to_annotate(save_path)
    
    def wordnet_example(self, synset, name, lemma):

        examples = synset.examples()
        sentences = []

        for sent in examples:
            sent = [Token(word = t.text, lemma=t.lemma_, pos=t.pos_, sense='-1') for t in self.nlp(sent)]
            for idx, t in enumerate(sent):
                if name == t.lemma:
                    t.sense = lemma
                    sent[idx] = t
                    sentences.append(Context(tokens = sent, index=idx, gloss = synset.definition()))
                    break

        return sentences
    
    def semcor_example(self, lemma):
        if lemma not in self.lemma2context:
            return []
        return self.lemma2context[lemma]
    
    def get_lemmas_to_annotate(self, save_path):
        for word, candidate_lemmas in self.lemma_generator():
            for lemma in candidate_lemmas:
                lemma.update({'word': word})
                self.all_lemmas.append(lemma)
        print(f"finish having all lemmas to annotate.")

        with open(os.path.join(save_path, f"lemma_{self.pos}.pickle"), 'wb') as f:
            pickle.dump(self.all_lemmas, f)
        print(f"Saved to {save_path} .")

    def lemma_generator(self):
        all_words = wn.all_lemma_names(pos = self.pos)
        for word in all_words:
            synsets = wn.synsets(word, pos = self.pos)
            if not (self.min_senses < len(synsets) < self.max_senses):
                continue
            candidate_lemmas = []
            for synset in synsets:
                lemma = [lemma.key() for lemma in synset.lemmas() if lemma.name() == word]
                if not len(lemma):
                    continue
                lemma = lemma[0]
                wn_examples = self.wordnet_example(synset, word, lemma)
                if not len(wn_examples) < self.min_wn_examples:
                    continue
                lemma_examples = self.semcor_example(lemma)
                if len(lemma_examples) >= self.min_semcor_examples_per_sense:
                    candidate_lemmas.append({'lemma': lemma, 'num_wn_examples': len(wn_examples), 'num_semcor_examples': len(lemma_examples)})
            if self.min_semcor_examples_per_word:
                num_semcor_per_word = sum([lemma['num_semcor_examples'] for lemma in candidate_lemmas])
                if num_semcor_per_word >= self.min_semcor_examples_per_sense*len(candidate_lemmas):
                    yield word, candidate_lemmas

if __name__ == '__main__':
    lemma_to_annotate('a')