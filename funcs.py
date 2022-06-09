from typing import Dict
import pandas as pd
from nltk.corpus import wordnet as wn
from sklearn.metrics import accuracy_score, f1_score, classification_report
import string
import nltk
from nltk import word_tokenize
from metaphor_identify_glue import tasks, task_to_keys
import datasets
import numpy as np
from lyc.utils import get_tokenizer

def add_metaphor_marker_with_meta_label_sequence(tokens, labels):
    assert len(tokens) == len(labels), f'{tokens} {labels}'
    if len(tokens) != len(labels):
        print(f'{tokens} {labels} **{len(tokens)-len(labels)}')
        return ''
    end_marker = 0
    for index, (token, label) in enumerate(zip(tokens, labels)):
        if label:
            if end_marker:
                continue
            if token.startswith('Ġ'):
                tokens[index] = token[0] + '<m>' + token[1:]
            else: 
                tokens[index] = '<m>' + token
            end_marker = 1
        else:
            if (token.startswith('Ġ') or token=='</s>') and end_marker:
                tokens[index] = '</m>' + tokens[index]
                end_marker=0
        # if token.startswith('Ġ'):
        #     if label: tokens[index] = token[0] + '<m>' + token[1:]
        #     if end_marker:
        #         tokens[index] = '</m>' + tokens[index]
        #         end_marker=0
        # else: 
        #     if label:tokens[index] = '<m>' + token
        # if label: end_marker = 1
    sentence = ''.join(tokens).replace('Ġ', ' ')
    return sentence

def get_gleu_error_case_with_meta_label(meta_type = 'vua', num_samples = 100):
    def process(x):
        results = {}
        results[col1] = add_metaphor_marker_with_meta_label_sequence(x[col1+f'_{meta_type}_token'], x[col1+f'_{meta_type}'])
        if col2 is not None:
            results[col2] = add_metaphor_marker_with_meta_label_sequence(x[col2+f'_{meta_type}_token'], x[col2+f'_{meta_type}'])
        return results

    for task in tasks:
        if task == 'mnli': continue
        col1, col2 = task_to_keys[task]
        ds = datasets.load_from_disk(f'glue/{meta_type}/{task}')
        if 'pred' not in ds.column_names:
            df = pd.read_csv(f'glue/val_results_with_metaphoricity/result_{task}.tsv', sep='\t')
            pred, gold, meta_label = df['prediction'], df['label'], df[f'{meta_type}_label']
            ds = ds.add_column('pred', pred)
            ds = ds.add_column('gold', gold)
            ds = ds.add_column(f'{meta_type}_label', meta_label)
            ds.save_to_disk(f'glue/{meta_type}_with_sentence/{task}')
        remove_cols = ['label', 'idx', f'{col1}_{meta_type}', f'{col1}_{meta_type}_token']
        if col2 is not None:
            remove_cols += [f'{col2}_{meta_type}', f'{col2}_{meta_type}_token']
        ds = ds.map(process, load_from_cache_file=False, remove_columns=remove_cols)
        df = ds.to_pandas()
        # df = df[ df['pred']==df['gold'] ]
        df = df[ df[f'{meta_type}_label']==0]
        num_samples = min(num_samples, len(df.index))
        samples = df.sample(n=num_samples, replace=False)
        samples.to_csv(f'glue/{meta_type}_test_without_meta/{task}.tsv', index=False, sep='\t')
        print(f'succesfully saved {task}.')

def produce_metaphoricity_for_glue():

    def compute_metaphorical_score(x, label_name, cols):
        col1, col2 = cols
        if col2 is not None:
            labels = np.concatenate([x[f'{col1}_{label_name}'], x[f'{col2}_{label_name}']])
        else:
            labels = x[f'{col1}_{label_name}']
        
        return sum(labels)/(len(labels)-2)

    for task in tasks:
        # if task == 'mnli': continue
        cols = task_to_keys[task]
        ds_moh = f'glue/moh/{task}'
        ds_vua = f'glue_mermaid/{task}'

        df_moh = datasets.load_from_disk(ds_moh).to_pandas()
        df_vua = datasets.load_from_disk(ds_vua).to_pandas()

        moh_labels = df_moh.apply(compute_metaphorical_score, axis=1, args=('moh', cols))
        vua_labels = df_vua.apply(compute_metaphorical_score, axis=1, args=('vua', cols))

        df_glue = pd.read_csv(f'glue/val_results_with_vua/vua_and_result_{task}.tsv', sep='\t')
        output_path = f'glue/val_results_with_metaphoricity/result_{task}.tsv'
        df_glue['vua_label'] = vua_labels
        df_glue['moh_label'] = moh_labels
        df_glue.to_csv(output_path, index=False, sep='\t')
        print(f'Svaed to {output_path}')


def load_moh_dataset(path):
    df = pd.read_csv(path, sep='\t')
    return df

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

def get_verb_metaphoricity_annotation(df):
    """
    The input df should be the metaphoricity table
    """

    sense = df['sense']
    label = df['class']
    prob = df['confidence']
    sentences = df['sentence']

    lemma = sense.apply(get_lemma)
    df = pd.concat([lemma, label, prob, sentences], axis=1)

    return df

def get_verb_metaphoricity_dict(df):
    """
    df should be the metaphoricity table.
    """
    df = df.set_index('sense')
    return df.to_dict('index')

def split_wsd_results_to_meta_and_non_meta(meta_dict):
    
    def assign_metaphoricity(x):
        gsense = x['gsense']
        if gsense in meta_dict:
            label, confidence = meta_dict[gsense]['class'], meta_dict[gsense]['confidence']
        else:
            label, confidence = None, None
        x['class'] = label
        x['confidence'] = confidence
        return x

    # eval_datasets = ['semeval2007', 'semeval2013', 'semeval2015', 'senseval2', 'senseval3']
    eval_datasets = ['moh']
    gold_paths = [f'wsd/{ds}.gold.key.txt' for ds in eval_datasets ]
    pred_paths = [f'wsd/{ds}_predictions.txt' for ds in eval_datasets ]

    all_pred_instances = []

    for g_f, p_f, ds in zip(gold_paths, pred_paths, eval_datasets):
        with open(g_f, 'r', encoding='utf-8') as g, open(p_f, 'r', encoding='utf-8') as p:
            for gline, pline in zip(g,p):
                gline, pline = gline.strip(), pline.strip()
                if not gline or not pline:
                    continue
                gid, gsense = gline.split(' ')[:2]
                pred_elements = pline.split(' ')

                if len(pred_elements)>2:
                    pid, psense, pvalue, _, gvalue = pred_elements
                else:
                    pid, psense = pred_elements

                assert gid == pid, "Glod and Pred data not been aligned."
                gid = ds + '.' + gid
                all_pred_instances.append({'id':gid, 'gsense':gsense, 'psense':psense, 'gvalue': gvalue, 'pvalue': pvalue})
    
    df = pd.DataFrame(all_pred_instances)
    df = df.apply(assign_metaphoricity, axis=1)

    return df, df[df['class']=='metaphorical'], df[df['class']=='literal']

def compute_meta_non_meta_acc_f1(df):
    literal = df[df['class']=='literal']
    meta = df[df['class']=='metaphorical']

    print('Literal Senses ....')
    report = classification_report(literal['gsense'], literal['psense'])
    print(report)

    print('\n\nMetaphorical Sense ...')
    report = classification_report(meta['gsense'], meta['psense'])
    print(report)

    print('\n\n ALL ... ')
    report = classification_report(df['gsense'], df['psense'], output_dict=True)
    print(report['accuracy'], report['macro avg'], report['weighted avg'])

def convert_moh_to_wsd_inference_input(df: pd.DataFrame, output_path, gold_output_path):
    with open(output_path, 'w', encoding='utf-8') as f, open(gold_output_path, 'w', encoding='utf-8') as gf:
        for index, line in df.iterrows():
            lemma = line['term']
            sense = line['sense']
            if pd.isna(sense):
                continue
            lemma_key = get_lemma(sense)
            sentence = line['sentence']
            sentence = sentence.replace('<b>', '-').replace('</b>', '-')
            tokens = word_tokenize(sentence)
            for token in tokens:
                if token.startswith('-') and token.endswith('-') and len(token)>2:
                    token = token.strip('-')
                    lemma_ = lemma
                    pos = 'v'
                    id = f'{index}.{lemma_key}'
                    label = lemma_key
                    gf.write(id + ' ' + label + '\n')
                else:
                    lemma_ = token
                    pos = ''
                    id = str(-1)
                    label = str(-1)
                f.write('\t'.join([token, lemma_, pos, id, label])+'\n')
            f.write('\n')

def show_moh_error_examples(meta_dict: Dict, wsd_result: pd.DataFrame, output_path):
    """
    The verb_table should be the moh verb annotation table;
    and the wsd_result should be the wsd metaphoricity table.

    ONLY supporting moh error showing till now.
    """

    output = []

    def filter(x):
        gsense = x['gsense']
        psense = x['psense']
        if gsense == psense:
            g_gloss = wn.lemma_from_key(gsense).synset().definition()
            p_gloss = wn.lemma_from_key(psense).synset().definition()
            sent = meta_dict[gsense]['sentence']
            output.append({'gsense': gsense, 'psense':psense, 'g_gloss':g_gloss, \
                'p_gloss':p_gloss, 'sentence':sent, 'g_class':meta_dict[gsense]['class'], \
                    'confidence': meta_dict[gsense]['confidence'], 'g_value': x['gvalue'], 'p_value':x['pvalue']})
    
    wsd_result.apply(filter, axis=1)
    df = pd.DataFrame(output)
    df.to_csv(output_path, index = False, sep='\t')

def show_inference_confidence_for_meta_and_literal(df):
    """
    df should be the metphoricity table.
    """

    meta_pvalue = df[df['class']=='metaphorical']['pvalue']
    literal_pvalue = df[df['class']=='literal']['pvalue']

    meta_pvalue = pd.to_numeric(meta_pvalue)
    literal_pvalue = pd.to_numeric(literal_pvalue)

    print(f'pvalue for metaphors: {meta_pvalue.mean()} \npvalue for literal: {literal_pvalue.mean()}')


if __name__ == '__main__':
    ## make verb metaphroicity table
    # df = load_moh_dataset('Metaphor-Emotion-Data-Files/Data-metaphoric-or-literal.txt')
    # df = get_verb_metaphoricity_annotation(df)
    # df.to_csv('Metaphor-Emotion-Data-Files/verb_metaphoricity.tsv', index=False, sep='\t')

    ## make wsd results with metaphoricity 
    # df = load_moh_dataset('Metaphor-Emotion-Data-Files/verb_metaphoricity.tsv')
    # meta_dict = get_verb_metaphoricity_dict(df)
    # df, _, _ = split_wsd_results_to_meta_and_non_meta(meta_dict)
    # df.to_csv('wsd/wsd.moh.metaphoricity.tsv', index=False, sep='\t')

    ## get metrics
    # df = load_moh_dataset('wsd/wsd.moh.metaphoricity.tsv')
    # compute_meta_non_meta_acc_f1(df)

    # ## generate wsd inference data
    # df = load_moh_dataset('Metaphor-Emotion-Data-Files/Data-metaphoric-or-literal.txt')
    # convert_moh_to_wsd_inference_input(df, 'Metaphor-Emotion-Data-Files/moh.wsd.inference.tsv', 'Metaphor-Emotion-Data-Files/moh.gold.key.txt')

    ## error analysis
    # df = load_moh_dataset('Metaphor-Emotion-Data-Files/verb_metaphoricity.tsv')
    # meta_dict = get_verb_metaphoricity_dict(df)
    # df = load_moh_dataset('wsd/wsd.moh.metaphoricity.tsv')
    # show_moh_error_examples(meta_dict, df, 'Metaphor-Emotion-Data-Files/moh.correct.tsv')

    ## Show pvalue for metaphor and literal
    # df = load_moh_dataset('wsd/wsd.moh.metaphoricity.tsv')
    # show_inference_confidence_for_meta_and_literal(df)

    ## produce glue results table with metaphroicity label score
    produce_metaphoricity_for_glue()

    ## produce glue error examples
    # get_gleu_error_case_with_meta_label()
