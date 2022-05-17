import sys
sys.path.append('/user/HS502/yl02706/mpa')
from lyc.model import simcse
from util import word2sentence, synset2sentence, Token, Context
from torch.utils.data import IterableDataset
import torch
from lyc.data import SimCSEDataSet, get_hf_ds_scripts_path
from lyc.utils import get_tokenizer, get_model
from lyc.train import get_base_hf_args
import random
import os
import pickle
import numpy as np
from typing import Union, List
from transformers import Trainer
from nltk.corpus import wordnet as wn
import datasets

class SenseCL(IterableDataset):
    def __init__(self, tokenizer, index_path = 'embeddings/index', max_steps = 300, batch_size = 32, min_synsets = 5, min_sents = 2, \
        max_length = 100):
        self.synset2sentence = synset2sentence(tokenizer, index_path=index_path)
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.min_synsets = min_synsets
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lemmatized2sentences = self._load_lemmatized2sentences(min_synsets, min_sents, index_path)

        self.lemmatized_list = list(self.lemmatized2sentences.keys())
        self._init_sampling_weight()

    def _load_lemmatized2sentences(self, min_synsets, min_sents, index_path):
        lemma2sentences_pkl = os.path.join(index_path, f'lemmatized2synsets_{min_synsets}_{min_sents}_{self.max_length}.pkl')
        if not os.path.exists(lemma2sentences_pkl):
            lemmatized2synsets = {}
            for sent, sent_encoding in zip(self.synset2sentence.sentences, self.synset2sentence.sentences_encoding):
                if len(sent_encoding['word_ids']) > self.max_length:
                    continue
                for token in sent:
                    if token.sense != '-1':
                        synset = wn.lemma_from_key(token.sense).synset().name()
                        lemmatized = token.lemma
                        if lemmatized not in lemmatized2synsets: lemmatized2synsets[lemmatized] = {}
                        if synset not in lemmatized2synsets[lemmatized]: lemmatized2synsets[lemmatized][synset] = 0
                        lemmatized2synsets[lemmatized][synset]+=1
            new_map = {}
            for lemmatized, synsets in lemmatized2synsets.items():
                synset_counter = 0
                for synset, count in synsets.items():
                    if count > min_sents: synset_counter+=1
                if synset_counter > min_synsets: new_map[lemmatized] = synsets
            
            with open(lemma2sentences_pkl, 'wb') as f:
                pickle.dump(new_map, f)
            
            return new_map
        
        with open(lemma2sentences_pkl, 'rb') as f:
            return pickle.load(f)
    
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
            selected_sentences = random.choices(self.synset2sentence(synset, max_length=self.max_length), k=2)
            sampled_sentences.extend(selected_sentences)
        label = np.arange(self.batch_size*2)
        label = (label+1)-(label%2)*2
        return sampled_sentences, label
    
    def preprocess_context(self, sentences):
        encoding = [] 
        idxs = []
        for sent in sentences:
            encoding.append(sent['encoding'])
            idxs.append(sent['idx'])
        
        encoding = self.tokenizer.pad(
            encoding,
            padding = True,
            max_length = self.max_length,
            return_tensors = 'pt'
        )

        return encoding, idxs
    
    def __iter__(self):
        count = 0
        while count < self.max_steps:
            sampled_sentences, label = self._sample_batch()
            encoding, new_idxs= self.preprocess_context(sampled_sentences)
            yield {'idxs': new_idxs, 'label': label, **encoding}
            count+=1
    
    def __len__(self):
        return self.max_steps

class FrameCL(IterableDataset):
    def __init__(self, tokenizer, data_path, max_length, max_steps, minimum_sents = 2, batch_size = 32):
        hfds_script = get_hf_ds_scripts_path('sesame')
        ds = datasets.load_dataset(hfds_script, data_dir=data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.minimum_sents = minimum_sents

        self.frame2encoding = {}
        self._prepare(ds)

    def _prepare(self, ds, ):
        ds = datasets.concatenate_datasets([ds['train'], ds['validation']])
        ignore_count = 0
        for line in ds:
            frame_tags = line['frame_tags']
            tokens = line['tokens']

            nonzeros = [(i, tag) for i, tag in enumerate(frame_tags) if tag]
            try:
                assert len(nonzeros) == 1, "multi frame tags per sentence"
            except:
                ignore_count +=1
                continue

            encoding = self.tokenizer(tokens, is_split_into_words = True)
            idxs = encoding.word_ids()
            if len(idxs) > self.max_length:
                continue
            tag = nonzeros[0][1]
            target_idx = nonzeros[0][0]

            new_idx = idxs.index(target_idx)
            if tag not in self.frame2encoding:
                self.frame2encoding[tag] = []
            self.frame2encoding[tag].append({'encoding': encoding, 'index': new_idx})
        
        print(f'{ignore_count} instances are ignored.')
        
        frame_sampling_weight = []
        self.frame_list = []
        for frame in self.frame2encoding:
            num_sents = len(self.frame2encoding[frame])
            if num_sents < self.minimum_sents:
                continue
            self.frame_list.append(frame)
            frame_sampling_weight.append(num_sents)
        self.frame_sampling_weight = np.array(frame_sampling_weight)/sum(frame_sampling_weight)
    
    def _sample_batch(self):
        sampled_frames = np.random.choice(self.frame_list, size=self.batch_size, replace=False, p=self.frame_sampling_weight)
        sample_encoding_1 = []
        sample_encoding_2 = []
        sample_idx_1 = []
        sample_idx_2 = []
        for frame in sampled_frames:
            sampled_sent = random.sample(self.frame2encoding[frame], k=2)
            sample_encoding_1.append(sampled_sent[0]['encoding'])
            sample_encoding_2.append(sampled_sent[1]['encoding'])
            sample_idx_1.append(sampled_sent[0]['index'])
            sample_idx_2.append(sampled_sent[1]['index'])
        
        encoding = self.tokenizer.pad(
            sample_encoding_1 + sample_encoding_2,
            padding = True,
            max_length = self.max_length,
            return_tensors = 'pt'
        )
        
        idx1=torch.arange(self.batch_size*2)[None, :]
        idx2=(idx1.T+self.batch_size)%(self.batch_size*2)
        label = torch.LongTensor(idx2).squeeze(-1)

        return encoding, sample_idx_1+sample_idx_2, label
    
    def __iter__(self):
        count = 0
        while count <= self.max_steps:
            batch, idxs, labels = self._sample_batch()
            yield {'idxs': idxs, 'label': labels, **batch}
            count += 1
    
    def __len__(self):
        return self.max_steps

if __name__ == '__main__':
    model_path, index_path, max_steps, save_path, max_length, data_type, = sys.argv[1:]
    max_length = int(max_length)
    max_steps = int(max_steps)
    assert data_type in ['sense', 'frame'], f"Not supported data type {data_type}"

    tokenizer = get_tokenizer(model_path, add_prefix_space=True)
    if data_type == 'sense':
        ds = SenseCL(tokenizer, index_path=index_path, max_steps = max_steps, max_length = max_length)
    elif data_type == 'frame':
        ds = FrameCL(tokenizer, index_path, max_length=max_length, max_steps=max_steps)

    args = get_base_hf_args(
        output_dir=save_path,
        train_batch_size=1,
        epochs=1,
        lr=5e-5,
        save_steps=200,
        save_strategy='steps',
        save_total_limit=5,
        group_by_length = False
    )

    # model = simcse(model_path, pooling_type='idx-last')
    model = get_model(simcse, model_path, pooling_type='idx-last', output_hidden_states = True)

    trainer = Trainer(
        args=args,
        model = model,
        train_dataset=ds,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model()