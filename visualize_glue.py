import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np
import datasets

tasks=['cola', 'sst2', 'mrpc', 'stsb', 'qqp', 'qnli', 'rte', 'wnli', 'mnli', ]

def compute_metric(df, task=None, threshold=0, model_type = 'vua', metric = accuracy_score):
    metaphors = df[df[f'{model_type}_label']>threshold]
    non_metaphors = df[df[f'{model_type}_label']<=threshold]

    meta = metric(metaphors['label'], metaphors['prediction'])
    literal = metric(non_metaphors['label'], non_metaphors['prediction'])
    all_ = metric(df['label'], df['prediction'])

    results = {}
    for k,v in meta.items():
        results[f'meta_{threshold}_{k}'] = v
    for k,v in meta.items():
        results[f'literal_{threshold}_{k}'] = v
    for k,v in meta.items():
        results[f'all_{threshold}_{k}'] = v
    
    return results

if __name__ == '__main__':
    # result_dir = 'glue/val_results_with_vua/'
    model_type = 'moh'
    result_dir = 'glue/val_results_with_metaphoricity/'

    dfs = { task: pd.read_csv(result_dir+f'result_{task}.tsv', sep='\t') for task in tasks }

    statistics = {}
    for task, df in dfs.items():
        if task not in statistics:
            statistics[task] = {}

        # metaphor ratio
        statistics[task]['token_ratio'] = df[f'{model_type}_label'].sum() / len(df.index)
        statistics[task]['sentence_ratio'] = (df[f'{model_type}_label']>0).sum() / len(df.index)
        statistics[task]['sentence_ratio_0.1'] = (df[f'{model_type}_label']>0.1).sum() / len(df.index)

        # if task == 'stsb':
        #     metric = datasets.load_metric('glue', 'stsb')
        #     statistics[task]['metaphor_acc_0'], statistics[task]['non_metaphor_acc_0'] = compute_acc(df, task=task, model_type = model_type, metric = metric)
        #     statistics[task]['metaphor_acc_0.1'], statistics[task]['non_metaphor_acc_0.1'] = compute_acc(df, task=task, threshold=0.1, model_type = model_type, metric = metric)
        #     break

        metric = datasets.load_metric('glue', task)
        # threshold 0
        results = compute_metric(df, task=task, model_type = model_type, metric = metric)
        statistics[task].update(results)
        # threshold 0.1
        results = compute_metric(df, task=task, threshold=0.1, model_type = model_type, metric = metric)
        statistics[task].update(results)
    
    statistics = pd.DataFrame.from_dict(statistics, orient='index')
    fig, axes = plt.subplots(4,1,figsize=(8,8))

    statistics[['token_ratio', 'sentence_ratio']].plot(ax=axes[0], style=['+','o'], yticks=np.linspace(0,1,6), markerfacecolor='none')
    statistics[['token_ratio', 'sentence_ratio_0.1']].plot(ax=axes[1], style=['+','o'], yticks=np.linspace(0,1,6), markerfacecolor='none')
    statistics[['metaphor_acc_0', 'non_metaphor_acc_0']].plot(ax=axes[2], style=['^','s'], yticks=np.linspace(0,1,6), markerfacecolor='none')
    statistics[['metaphor_acc_0.1', 'non_metaphor_acc_0.1']].plot(ax=axes[3], style=['^','s'], yticks=np.linspace(0,1,6), markerfacecolor='none')

    axes[0].set_ylabel('Metaphor ratio')
    axes[1].set_ylabel('Metaphor ratio \n threshold set to 0.1')
    axes[2].set_ylabel('Accuracy for \n metaphor and \n non_metaphor examples')
    axes[3].set_ylabel('Accuracy where \n metaphor threshold \n is set to 0.1')


    # statistics.plot(subplots=True, style=['*', '+', 'o', 's', '^', '+'])

    plt.show()