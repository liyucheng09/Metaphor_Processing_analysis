import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import numpy as np

tasks=['cola', 'sst2', 'mrpc', 'stsb', 'qqp', 'qnli', 'rte', 'wnli']

def compute_acc(df, task=None, threshold=0):
    metaphors = df[df['vua_label']>threshold]
    non_metaphors = df[df['vua_label']<=threshold]
    return accuracy_score(metaphors['label'], metaphors['prediction']), accuracy_score(non_metaphors['label'], non_metaphors['prediction'])

if __name__ == '__main__':
    # result_dir = 'glue/val_results_with_vua/'
    model_type = 'moh'
    result_dir = 'glue/val_results_with_metaphoricity/'

    dfs = { task: pd.read_csv(result_dir+f'result_{task}.tsv', sep='\t') for task in tasks }

    statistics = {}
    for task, df in dfs.items():
        if task == 'stsb':
            continue

        if task not in statistics:
            statistics[task] = {}

        # metaphor ratio
        statistics[task]['token_ratio'] = df[f'{model_type}_label'].sum() / len(df.index)
        statistics[task]['sentence_ratio'] = (df[f'{model_type}_label']>0).sum() / len(df.index)

        # acc threshold 0
        statistics[task]['metaphor_acc_0'], statistics[task]['non_metaphor_acc_0'] = compute_acc(df, task=task)

        # acc threshold 0.1
        statistics[task]['metaphor_acc_0.1'], statistics[task]['non_metaphor_acc_0.1'] = compute_acc(df, task=task, threshold=0.1)
    
    statistics = pd.DataFrame.from_dict(statistics, orient='index')
    fig, axes = plt.subplots(3,1,figsize=(8,8))

    statistics[['token_ratio', 'sentence_ratio']].plot(ax=axes[0], style=['+','o'], yticks=np.linspace(0,1,6), markerfacecolor='none')
    statistics[['metaphor_acc_0', 'non_metaphor_acc_0']].plot(ax=axes[1], style=['^','s'], yticks=np.linspace(0,1,6), markerfacecolor='none')
    statistics[['metaphor_acc_0.1', 'non_metaphor_acc_0.1']].plot(ax=axes[2], style=['^','s'], yticks=np.linspace(0,1,6))

    axes[0].set_ylabel('Metaphor ratio')
    axes[1].set_ylabel('Accuracy for \n metaphor and \n non_metaphor examples')
    axes[2].set_ylabel('Accuracy where \n metaphor threshold \n is set to 0.1')


    # statistics.plot(subplots=True, style=['*', '+', 'o', 's', '^', '+'])

    plt.show()