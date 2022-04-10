from lyc.model import simcse
from util import word2sentence, synset2sentence, Token, Context
from torch.utils.data import IterableDataset
from lyc.data import SimCSEDataSet
from lyc.utils import get_tokenizer, get_model
from lyc.train import get_base_hf_args
import random
import os
import pickle
import numpy as np
import sys
from transformers import Trainer

class SenseCL(IterableDataset):
    def __init__(self, tokenizer, index_path = 'embeddings/index', max_steps = 300, batch_size = 32, min_synsets = 5, min_sents = 2, \
        max_length = 100):
        self.lemmatized2sentences = self._load_lemmatized2sentences(min_synsets, min_sents, index_path)
        self.synset2sentence = synset2sentence(index_path=index_path)
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.min_synsets = min_synsets
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.lemmatized_list = list(self.lemmatized2sentences.keys())
        self._init_sampling_weight()
    
    def _load_lemmatized2sentences(self, min_synsets, min_sents, index_path):
        assert os.path.exists(f'{index_path}/lemmatized2sentences_semcor.pkl')
        with open('embeddings/index/lemmatized2sentences_semcor.pkl', 'rb') as f:
            lemmatized2sentences = pickle.load(f)
        new_map = {}
        for lemma, synset2sents in lemmatized2sentences.items():
            synset_counter = 0
            for synset, sents in synset2sents.items():
                if len(sents) > min_sents: synset_counter+=1
            if synset_counter > min_synsets: new_map[lemma] = synset2sents
        return new_map
    
    def _init_sampling_weight(self, upper_bound = 1000, lower_bound = 100):
        synset_frequencies = self.synset2sentence.synsets_frequencies()
        lemmatized_frequencies = [sum([synset_frequencies[synset] for synset in self.lemmatized2sentences[lemmatized]]) for lemmatized in self.lemmatized_list]
        lemmatized_frequencies = np.clip(lemmatized_frequencies, lower_bound, upper_bound)
        self.lemmatized_frequencies = lemmatized_frequencies
        self.synset_frequencies = synset_frequencies
    
    def _sample_batch(self):
        """Return a batch of Context objects.
        """
        selected_lemmatizeds = random.choices(self.lemmatized_list, weights=self.lemmatized_frequencies, k=self.batch_size)
        selected_synsets = set()
        for lemmatized in selected_lemmatizeds:
            if len(selected_synsets) > self.batch_size: break
            selected_synsets = selected_synsets.union(set(self.lemmatized2sentences[lemmatized].keys()))
        selected_synsets = list(selected_synsets)[:self.batch_size]
        sampled_sentences = []
        for synset in selected_synsets:
            selected_sentences = random.choices(self.synset2sentence(synset), k=2)
            sampled_sentences.extend(selected_sentences)
        return self.synset2sentence.realized_to_context(sampled_sentences)
    
    def preprocess_context(self, contexts: list[Context]):
        sents = []
        idxs = []
        senses = []
        new_idxs = []

        for cont in contexts:
            idx = cont.index
            target_sense = cont.tokens[idx].sense
            sent = [t.word for t in cont.tokens]
            
            sents.append(sent)
            idxs.append(idx)
            senses.append(target_sense)
        
        output = self.tokenizer(sents, is_split_into_words = True, max_length = self.max_length, padding=True, truncation = True, return_tensors = 'pt')
        for i in range(len(sents)):
            new_idx = output.word_ids(i)
            new_idx = new_idx.index(idxs[i])
            new_idxs.append(new_idx)
        
        # new_idxs = torch.LongTensor(new_idxs)
        
        return output, new_idxs, senses
    
    def __iter__(self):
        count = 0
        while count < self.max_steps:
            sampled_sentences = self._sample_batch()
            encoding, new_idxs, senses = self.preprocess_context(sampled_sentences)
            yield {'idxs': new_idxs, **encoding}
            count+=1


if __name__ == '__main__':
    model_path, index_path, max_steps, = sys.argv[1:]

    tokenizer = get_tokenizer(model_path, add_prefix_space=True)
    ds = SenseCL(tokenizer, index_path=index_path, max_steps = max_steps)

    args = get_base_hf_args(
        output_dir='checkpoints/senseCL',
        train_batch_size=1,
        epochs=1,
        lr=5e-5,
        save_steps=100,
        save_strategy='steps',
        save_total_limit=5
    )

    model = simcse(model_path, pooling_type='idx-last')

    trainer = Trainer(
        args=args,
        model = model,
        train_dataset=ds,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model()