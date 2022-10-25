import pandas as pd
import datasets

tasks = ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli', 'boolq']

if __name__ == '__main__':
    
    for task in tasks:
        unmask_file_path = f'glue_val/results/vua_and_result_{task}.tsv'
        masked_file_path = f'glue_val/results/vua_and_result_{task}_masked.tsv'
        save_file_path = f'glue_val/results/vua_and_result_{task}_critical_metaphors.tsv'

        try:
            unmask_df = pd.read_csv(unmask_file_path, sep='\t', index_col='index')
            masked_df = pd.read_csv(masked_file_path, sep='\t', index_col='index')
        except FileNotFoundError:
            print(f'Files for {task} were not found.')
            continue

        critical_metaphors = unmask_df[unmask_df['prediction']!=masked_df['prediction']]
        critical_metaphors = critical_metaphors[critical_metaphors['vua_label']>0]
        critical_metaphors.to_csv(save_file_path, sep='\t', index=False)

        metric = datasets.load_metric('glue', task)
        print(f'Finished writing to {save_file_path}')
        print(metric.compute(predictions=critical_metaphors['prediction'], references=critical_metaphors['label']))
        print('\n--\n')


