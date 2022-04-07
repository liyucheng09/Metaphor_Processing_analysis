from lyc.model import simcse
from util import word2sentence, synset2sentence
from torch.utils.data import IterableDataset
from lyc.data import SimCSEDataSet
import random
import os
import pickle

class SenseCL(IterableDataset):
    def __init__(self, max_steps = 300, batch_size = 32, min_synsets = 5, min_sents = 2):
        self.lemmatized2sentences = self._load_lemmatized2sentences(min_synsets, min_sents)
        self.synset2sentence = synset2sentence()
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.min_synsets = min_synsets
    
    def _load_lemmatized2sentences(self, min_synsets, min_sents, index_path = 'embeddings/index'):
        assert os.path.exists(f'{index_path}/lemmatized2sentences.pkl')
        with open('embeddings/index/lemmatized2sentences.pkl', 'rb') as f:
            lemmatized2sentences = pickle.load(f)
        new_map = {}
        for lemma, synset2sents in lemmatized2sentences.items():
            synset_counter = 0
            for synset, sents in synset2sents.items():
                if len(sents) > min_sents: synset_counter+=1
            if synset_counter > min_synsets: new_map[lemma] = synset2sents
        return new_map
    
    def _sampling_weight(self):
    
    def __iter__(self):
        count = 0
        while count < self.max_steps:
            lemmatized = random.choice(self.lemmatized_list)
            synsets2sents = self.lemmatized2sentences(lemmatized)


if __name__ == '__main__':
    
