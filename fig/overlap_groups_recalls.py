import pandas as pd
import numpy as np
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt

if __name__ == '__main__':
    mi_results = 'Hard/mi/token_on.csv'
    hmc_path = 'embeddings/results/senseval3_instances_ChalMC.tsv'

    hmc = pd.read_csv(hmc_path, sep='\t')
    df = pd.read_csv(mi_results, sep='\t', names=['prediction', 'label', 'target'])

    df = df.dropna()

    df['overlap_score'] = hmc['overlap_score']
    df = df[df['label']==1]
    df['overlap_score'] = df['overlap_score'].astype(float)

    bins = np.arange(-0.001,0.8,0.2)
    df['overlap_group'] = pd.cut(df['overlap_score'], bins, labels = [1,2,3,4])
    
    x = []
    y = []
    for group_label, group in df.groupby('overlap_group'):
        x.append(group_label)
        recall = recall_score(group['label'], group['prediction'])
        y.append(recall)
        print(x,y)
    
    plt.figure(figsize=(4,6), ) #dpi=80,
    plt.plot(x, y, marker = '^')
    plt.yticks(np.arange(0.5, 0.71, 0.05))
    plt.xticks(np.arange(1, 5, 1), labels = ['0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8'])
    plt.xlabel('Overlap Ratio')
    plt.grid(axis='y')

    plt.show()