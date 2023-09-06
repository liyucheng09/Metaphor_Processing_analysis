import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

tasks = ['sst2', 'mrpc', 'qqp', 'qnli', 'rte', 'wnli', 'mnli']

if __name__ == '__main__':
    
    model_type = 'vua'
    result_dir = 'glue_val/all_mask_results/'

    dfs = { task: pd.read_csv(result_dir+f'vua_and_result_{task}.tsv', sep='\t') for task in tasks }

    statistics = {}
    for task, df in dfs.items():
        if task not in statistics:
            statistics[task] = {}
        
        statistics[task]['token_ratio'] = df[f'{model_type}_label'].sum() / len(df.index)
        statistics[task]['sentence_ratio'] = (df[f'{model_type}_label']>0).sum() / len(df.index)
        statistics[task]['sentence_ratio_0.1'] = (df[f'{model_type}_label']>0.1).sum() / len(df.index)

        metaphors = df[df[f'{model_type}_label']>0]
        non_metaphors = df[df[f'{model_type}_label']<=0]

        meta = accuracy_score(metaphors['label'], metaphors['prediction'])
        literal = accuracy_score(non_metaphors['label'], non_metaphors['prediction'])

        statistics[task]['meta_acc'] = meta
        statistics[task]['literal_acc'] = literal

    statistics['mt'] = {}
    statistics['mt']['meta_acc'] = 0.92
    statistics['mt']['literal_acc'] = 0.95
    statistics = pd.DataFrame.from_dict(statistics, orient='index')
    statistics[['meta_acc', 'literal_acc']].plot(style=['s', '^'], yticks=np.linspace(0,1,6), alpha = 0.7, figsize = (6,2.7),)
    plt.legend(['Accuracy for samples with metaphor', 'Accuracy for literal samples'])
    plt.show()