import pandas as pd
from nltk.corpus import wordnet as wn
from sklearn.metrics import accuracy_score, f1_score, classification_report

def load_moh_dataset(path):
    df = pd.read_csv(path, sep='\t')
    return df

def get_verb_metaphoricity_annotation(df):
    """
    The input df should be the metaphoricity table
    """
    
    def get_lemma(x):
        try:
            lemma, pos, sense_id = x.split('#')
        except:
            return None
        sense_id = int(sense_id)
        lemma_str = f'{lemma}.{pos}.{sense_id:02d}.{lemma}'
        return wn.lemma(lemma_str).key()

    sense = df['sense']
    label = df['class']
    prob = df['confidence']

    lemma = sense.apply(get_lemma)
    df = pd.concat([lemma, label, prob], axis=1)

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

    eval_datasets = ['semeval2007', 'semeval2013', 'semeval2015', 'senseval2', 'senseval3']
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
                pid, psense = pline.split(' ')
                assert gid == pid, "Gloden and Pred data not been aligned."
                gid = ds + '.' + gid
                all_pred_instances.append({'id':gid, 'gsense':gsense, 'psense':psense})
    
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


if __name__ == '__main__':
    ## make verb metaphroicity table
    # df = load_moh_dataset('Metaphor-Emotion-Data-Files/Data-metaphoric-or-literal.txt')
    # df = get_verb_metaphoricity_annotation(df)
    # df.to_csv('verb_metaphoricity.tsv', index=False, sep='\t')

    ## make wsd results with metaphoricity 
    # df = load_moh_dataset('Metaphor-Emotion-Data-Files/verb_metaphoricity.tsv')
    # meta_dict = get_verb_metaphoricity_dict(df)
    # df, _, _ = split_wsd_results_to_meta_and_non_meta(meta_dict)
    # df.to_csv('wsd/wsd.metaphoricity.tsv', index=False, sep='\t')

    ## get metrics
    df = load_moh_dataset('wsd/wsd.metaphoricity.tsv')
    compute_meta_non_meta_acc_f1(df)