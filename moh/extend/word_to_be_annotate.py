from nltk.corpus import wordnet as wn
import spacy
from embeddings.util import *
import pickle
import os
import pandas as pd

class lemma_to_annotate:

    def __init__(self, pos, min_senses = 3, max_senses = 10, min_wn_examples = 1, min_semcor_examples_per_sense = 3, min_semcor_examples_per_word = True,
                    save_path = 'moh/extend/'):
        self.nlp = spacy.load('en_core_web_sm', disable=["parser", "ner"])
        with open('embeddings/index/lemma2instances_ufsac.pkl', 'rb') as f:
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
                    sentences.append(Context(tokens = sent, index=idx, gloss = synset.definition(), sense_list=[lemma]))
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
        all_synsets = []
        for word in all_words:
            synsets = wn.synsets(word, pos = self.pos)
            if not (self.min_senses < len(synsets) < self.max_senses):
                continue
            all_synsets.extend(synsets)
            candidate_lemmas = []
            for synset in synsets:
                lemma = [lemma.key() for lemma in synset.lemmas() if lemma.name() == word]
                if not len(lemma):
                    continue
                lemma = lemma[0]
                wn_examples = self.wordnet_example(synset, word, lemma)
                if not len(wn_examples) >= self.min_wn_examples:
                    continue
                ufsac_lemma = lemma if '%5' not in lemma else lemma.replace('%5', '%3')
                lemma_examples = self.semcor_example(ufsac_lemma)
                if len(lemma_examples) >= self.min_semcor_examples_per_sense:
                    candidate_lemmas.append({'lemma': lemma, 'num_wn_examples': len(wn_examples), 'num_semcor_examples': len(lemma_examples)})
            if self.min_semcor_examples_per_word:
                num_semcor_per_word = sum([lemma['num_semcor_examples'] for lemma in candidate_lemmas])
                if num_semcor_per_word >= self.min_semcor_examples_per_sense*len(candidate_lemmas):
                    yield word, candidate_lemmas
    
    def annotation_forms(self, save_path, minimum_num_senses = 4):
        output_path = os.path.join(save_path, f"annotation_forms_{self.pos}.tsv")
        df = pd.DataFrame(self.all_lemmas)

        word_dfs = []
        for word, group in df.groupby('word'):
            if len(group) <= minimum_num_senses:
                continue
            word_dfs.append(group)
        df = pd.concat(word_dfs)

        def get_example_sentence(x):
            lemma = wn.lemma_from_key(x['lemma'])
            synset = lemma.synset()
            word = wn.lemma_from_key(x['lemma']).name()
            examples = self.wordnet_example(synset, word, lemma)

            return str(examples[0])
        
        def get_gloss(x):
            lemma = wn.lemma_from_key(x['lemma'])
            synset = lemma.synset()
            return synset.definition()

        df['gloss'] = df.apply(get_gloss, axis=1)
        df['example_sentence'] = df.apply(get_example_sentence, axis=1)

        df = df.drop(columns=['num_wn_examples', 'num_semcor_examples'])
        df.to_csv(output_path, sep='\t', index=False)
        print(f"Saved to {output_path} .")
                
def words_to_annotate(pos, minimum_num_senses = 4):
    with open(f'moh/extend/lemma_{pos}.pickle', 'rb') as f:
        lemmas_to_annotate = pickle.load(f)
    
    with open(f'moh/extend/words_to_annotate_pos_{pos}.tsv', 'w') as f:
        df = pd.DataFrame(lemmas_to_annotate)
        for word, group in df.groupby('word'):
            if len(group) <= minimum_num_senses:
                continue
            f.write(f'{word}\t{len(group)}\t')
            f.write('\t'.join(group['lemma']))
            f.write('\n')
    
    print(f'Write to moh/extend/words_to_annotate_pos_{pos}.tsv')

if __name__ == '__main__':
    l2a = lemma_to_annotate('a')
    l2a.annotation_forms(save_path='moh/extend')
    # l2a = words_to_annotate('a')
