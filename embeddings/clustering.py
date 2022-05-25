import sys
sys.path.append('/user/HS502/yl02706/mpa')
from lyc.model import simcse
from lyc.utils import get_tokenizer, get_model
from lyc.data import SentenceDataset
from lyc.visualize import plotDimensionReduction
import pandas as pd
from util import word2lemmas, sense, Token, Context, word2sentence
from main import SenseEmbedding
import numpy as np
import os
from nltk.corpus import wordnet as wn
from collections import Counter

def compute_overlapping(vecs, lemmas, metaphorical_lemmas):
    """计算每个sense和其余sense之间的重叠程度。
    假设sense A共有N个contexts，那么我们令sense A的每个点寻找距离起其最近的N个邻居。
    将这N个邻居中sense A的比例，作为sense A与其余sense重合度的一种度量。

    Args:
        vecs (np.array): list of representation
        lemmas (np.array): list of lemma strings

    Returns:
        overlapping (dict): {lemma: overlap(__float__) }
    """
    num_vecs = vecs.shape[0]
    sims = np.matmul(vecs, vecs.T) - np.eye(num_vecs)*100
    sort_idxs = sims.argsort(axis=1)

    overlapping = {}
    for lemma in set(lemmas):
        xidx = (lemmas == lemma)
        idxs_of_lemma = xidx.nonzero()[0]
        num_of_context = xidx.sum()
        lemma_knn = sort_idxs[xidx][:, -(num_of_context-1):]
        overlap_mask = np.isin(lemma_knn, idxs_of_lemma)
        total = num_of_context * (num_of_context-1)

        overlap_score = overlap_mask.sum()/total

        noises = np.where(overlap_mask, -1, lemma_knn)
        knns = Counter(noises.reshape(-1))
        knns.pop(-1, None)
        top_k_noises = knns.most_common(num_of_context)
        idxs_of_noises = np.array([n[0] for n in top_k_noises])

        meta_idx = np.isin(lemmas, metaphorical_lemmas)
        idxs_of_meta = meta_idx.nonzero()[0]
        meta_mask = np.isin(lemma_knn, idxs_of_meta)
        non_meta_noises = np.where(meta_mask, -1, noises)
        knns = Counter(non_meta_noises.reshape(-1))
        knns.pop(-1, None)
        top_k_noises = knns.most_common(num_of_context)
        idxs_of_non_meta_noises = np.array([n[0] for n in top_k_noises])

        overlapping[lemma] = {'overlap_score': overlap_score, 'idxs': idxs_of_lemma, \
            'idxs_of_noises': idxs_of_noises, 'idxs_of_non_meta_noises': idxs_of_non_meta_noises}
    
    return overlapping

def instance_level_overlapping(vecs, lemmas, metaphorical_lemmas):
    """计算每个instance和其余instances之间的距离，观察instance周围的样例是否属于自己的sense。
    假设instance a 所属的sense A 共有N个contexts，那么我们寻找距离a最近的N个邻居。
    将这N个邻居中sense A的比例，作为instance a与其余sense重合度的一种度量。

    Args:
        vecs (np.array): list of representation
        lemmas (np.array): list of lemma strings

    Returns:
        overlapping (dict): {lemma: overlap(__float__) }
    """
    num_vecs = vecs.shape[0]
    sims = np.matmul(vecs, vecs.T) - np.eye(num_vecs)*100
    sort_idxs = sims.argsort(axis=1)

    overlapping = {}
    for lemma in set(lemmas):
        xidx = (lemmas == lemma)
        idxs_of_lemma = xidx.nonzero()[0]
        num_of_context = xidx.sum()
        lemma_knn = sort_idxs[xidx][:, -(num_of_context-1):]
        overlap_mask = np.isin(lemma_knn, idxs_of_lemma)

        overlap_score = overlap_mask.sum(axis=1)/(num_of_context-1)
        # hard_idxs = idxs_of_lemma[overlap_score > threshold_for_overlap]
        
        noises = np.where(overlap_mask, -1, lemma_knn)

        meta_idx = np.isin(lemmas, metaphorical_lemmas)
        idxs_of_meta = meta_idx.nonzero()[0]
        meta_mask = np.isin(lemma_knn, idxs_of_meta)

        non_meta_noises = np.where(meta_mask, -1, noises)
        # non_meta_noises = non_meta_noises[hard_idxs]

        most_ambiguouses = []
        for each_hard in non_meta_noises:
            try:
                most_ambiguous = (each_hard!=-1).nonzero()[0][-1]
                most_ambiguouses.append(each_hard[most_ambiguous])
            except:
                most_ambiguouses.append(None)

        overlapping[lemma] = []

        for idx, score, most_ambiguous in zip(idxs_of_lemma, overlap_score, most_ambiguouses):
            overlapping[lemma].append((idx, score, most_ambiguous))
    
    return overlapping

def hard_metaphor_corpus(corpus_file, metaphorical_sense_file, non_meta_sense_file,\
        word, lemma, overlap, contexts, is_metaphorical, gloss):

    overlap_score = overlap['overlap_score']
    idxs = [str(i) for i in  overlap['idxs'].tolist()]
    idxs_of_noise = [ str(i) for i in overlap['idxs_of_noises'].tolist()]
    idxs_of_non_meta_noise = [ str(i) for i in overlap['idxs_of_non_meta_noises'].tolist()]
    num_sents = lemma_counter[lemma]

    if is_metaphorical:
        lemma = '*' + lemma
        metaphorical_sense_file.write(f'{word}\t{lemma}\t{overlap_score}\t{gloss}\t{num_sents}\t{",".join(idxs)}\t{",".join(idxs_of_noise)}\t{",".join(idxs_of_non_meta_noise)}\n')
    else:
        non_meta_sense_file.write(f'{word}\t{lemma}\t{overlap_score}\t{gloss}\t{num_sents}\t{",".join(idxs)}\t{",".join(idxs_of_noise)}\t{",".join(idxs_of_non_meta_noise)}\n')

    if overlap_score < threshold_for_overlap and num_sents > threshold_for_num_sent and is_metaphorical:
        for i in idxs:
            sent = contexts[i]
            corpus_file.write(
                f"{word}\t{lemma}\t{gloss}\tmetaphorical\t{overlap_score}\t{repr(sent)}\t{sent.index}\n"
            )
        for i in idxs_non_meta_noises:
            sent = contexts[i]
            noise_lemma = sent.tokens[sent.index].sense
            corpus_file.write(
                f"{word}\t{noise_lemma}\t{lemma2gloss(noise_lemma)}\tliteral\t_\t{repr(sent)}\t{sent.index}\n"
            )
    print(f"write {word}-{lemma} to hard metaphor corpus!")

def instance_level_metaphor_corpus(corpus_file, word, lemma, overlap, contexts, gloss):
    for idx, score, most_ambiguous in overlap:
        num_sents = lemma_counter[lemma]

        if score < threshold_for_overlap and num_sents > threshold_for_num_sent:
            sent = contexts[idx]
            corpus_file.write(
                f"{word}\t{'*' + lemma}\t{gloss}\tmetaphorical\t{score}\t{repr(sent)}\t{sent.index}\n"
            )
            if most_ambiguous is not None:
                ambiguous_sent = contexts[most_ambiguous]
                noise_lemma = ambiguous_sent.tokens[ambiguous_sent.index].sense
                corpus_file.write(
                    f"{word}\t{noise_lemma}\t{lemma2gloss(noise_lemma)}\tliteral\t_\t{repr(ambiguous_sent)}\t{ambiguous_sent.index}\n"
                )
    print(f"write {word} to hard metaphor corpus!")

def sense_merge(contexts, sense_merging_mapping):
    for index, sent in enumerate(contexts):
        lemma = sent.tokens[sent.index].sense
        if lemma in sense_merging_mapping:
            sent.tokens[sent.index].sense = sense_merging_mapping[lemma]
            contexts[index] = sent
    return contexts

if __name__ == '__main__':
    cwd, max_length, model_path, pool, source, threshold_for_num_sent, threshold_for_overlap, level, = sys.argv[1:]
    assert level in ['instances', 'senses'], f"Not supported level {level}."
    threshold_for_num_sent = int(threshold_for_num_sent)
    threshold_for_overlap = float(threshold_for_overlap)
    max_length = int(max_length)

    index_path = os.path.join(cwd, 'embeddings/index')

    tokenizer = get_tokenizer('roberta-base', add_prefix_space=True)
    model = SenseEmbedding(model_path, add_prefix_space = True, pool = pool, max_length=max_length, output_hidden_states = True)
    word2sentence = word2sentence(source, tokenizer, index_path=index_path)

    if source == 'semcor':
        words = word2sentence.word2lemmas.moh_word2lemmas
        lemma2gloss = lambda lemma: wn.lemma_from_key(lemma).synset().definition()
    elif source == 'senseval3':
        words = word2sentence.word2lemmas.senseval3_word2lemmas
        glosses = {s.lemma : s.gloss for word, senses in words.items() for s in senses }
        lemma2gloss = glosses.get
        sense_merging_mapping = word2sentence.word2lemmas.senseval3_sense_merging

    # model = get_model(simcse, model_path, pool = pool, output_hidden_states = True)
    if level == 'senses':
        meta_sense_output_path = os.path.join(cwd, f'embeddings/results/{source}_meta_results.tsv')
        non_meta_sense_output_path = os.path.join(cwd, f'embeddings/results/{source}_non_meta_results.tsv')
        meta_sense_output_file = open(meta_sense_output_path, 'w')
        non_meta_sense_output_file = open(non_meta_sense_output_path, 'w')
        meta_sense_output_file.write('word\tlemma\toverlap_score\tgloss\tnum_sent\tidxs\tnoises_idxs\tnon_meta_noises\n')
        non_meta_sense_output_file.write('word\tlemma\toverlap_score\tgloss\tnum_sent\tidxs\tnoises_idxs\tnon_meta_noises\n')

    corpus_output_path = os.path.join(cwd, f'embeddings/results/{source}_{level}_ChalMC.tsv')
    corpus_output_file = open(corpus_output_path, 'w')
    corpus_output_file.write('word\tlemma\tgloss\tlabel\toverlap_score\tcontext\ttarget_index\n')
    for word in words:
        contexts = word2sentence(word, minimum = 1, max_length = max_length)
        contexts = sense_merge(contexts, sense_merging_mapping)
        # contexts = contexts[:100]
        if source == 'senseval3':
            contexts = [sent for sent in contexts if not ';' in sent.tokens[sent.index].sense]
        if not contexts:
            print(f'{word} do not have enough contexts!')
            continue
        lemmas = [ cont.tokens[cont.index].sense for cont in contexts]
        lemma_counter = Counter(lemmas)
        lemmas = np.array(lemmas)
        metaphorical_senses = [ sense.lemma for sense in words[word] if sense.label == 'metaphorical']

        vecs = model.get_embeddings(contexts)
        overlapping = compute_overlapping(vecs, lemmas, metaphorical_senses) if level == 'senses' else instance_level_overlapping(vecs, lemmas, metaphorical_senses)

        # output_overlapping_path = os.path.join(cwd, 'embeddings/overlapping', f'{model_id}_{word}.result')
        # model_id = os.path.basename(model_path)
        # img_path = os.path.join(cwd, 'embeddings/imgs/clustering', f'{source}_{model_id}_{word}.png')
        
        # overlap_path = open(output_overlapping_path, 'w', encoding='utf-8')
        # overlap_path.write(f'lemma\toverlap_score\tgloss\tnum_sent\n')
        for lemma, overlap in overlapping.items():
            gloss = lemma2gloss(lemma)
            is_metaphorical = lemma in metaphorical_senses

            if level == 'senses':
                hard_metaphor_corpus(corpus_output_file, meta_sense_output_file, non_meta_sense_output_file, \
                    word, lemma, overlap, contexts, is_metaphorical, gloss)
            elif level == 'instances' and is_metaphorical:
                instance_level_metaphor_corpus(corpus_output_file, word, lemma, overlap, contexts, gloss)
            # overlap_path.write(f'{lemma}\t{overlap}\t{gloss}\t{lemma_counter[lemma]}\t{",".join(idxs)}\t{",".join(idxs_of_noise)}\n')

        # overlap_path.close()
        # plotDimensionReduction(vecs, lemmas, img_path)
        print(f'{word} process finished!')
    
    if level == 'senses':
        meta_sense_output_file.close()
        non_meta_sense_output_file.close()
    corpus_output_file.close()