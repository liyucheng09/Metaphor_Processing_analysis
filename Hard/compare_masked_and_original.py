import pandas as pd
import datasets
from glob import glob

tasks = ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli', 'boolq']

def main(task):
    unmask_file_path = f'/vol/research/nlg/mpa/glue/results/vua_and_result_{task}.tsv'
    masked_files = glob(f'/vol/research/nlg/mpa/glue/results/vua_and_result_{task}_masked_*.tsv')
    save_file_path = f'/vol/research/nlg/mpa/glue/results/vua_and_result_{task}_critical_metaphors.tsv'
    
    mask_dfs = []
    if not masked_files:
        return
    for mask_file in masked_files:
        masked_df = pd.read_csv(mask_file, sep='\t', index_col='index')
        mask_dfs.append(masked_df)
    
    number_of_dfs = len(mask_dfs)
    unmask_df = pd.read_csv(unmask_file_path, sep='\t', index_col='index')
    
    label = unmask_df['prediction']*number_of_dfs
    preds = mask_dfs[0]['prediction']
    for df in mask_dfs:
        preds+=df['prediction']

    critical_metaphors = unmask_df[label!=preds]
    critical_metaphors = critical_metaphors[critical_metaphors['vua_label']>0]
    critical_metaphors.to_csv(save_file_path, sep='\t', index=False)

    metric = datasets.load_metric('glue', task)
    print(f'Finished writing to {save_file_path}')
    print(metric.compute(predictions=critical_metaphors['prediction'], references=critical_metaphors['label']))
    print(f'Count of critical metaphors: {len(critical_metaphors.index)}, ratio in all eval dataset: {len(critical_metaphors.index)/len(unmask_df.index)}')
    print('\n--\n')

if __name__ == '__main__':
    
    for task in tasks:
        main(task)


