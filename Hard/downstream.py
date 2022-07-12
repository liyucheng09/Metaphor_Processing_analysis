import datasets
import pandas as pd
import sys

task_to_cols = {
    'nli': {'Hypothesis' : 'sentence1', 'Claim': 'sentence2', 'Answer': 'label'},
    'qa': {'Question': 'question', 'context': 'passage', 'Answer': 'label'}
}

def load_dataset(data_type, data_path):
    """load metaphor dataset with downstream task annotations.
    Task `nli` should be test via the `MRPC` model.
    Task `qa` is a binary question answering problem with context. Should be test via `BoolQ` model from `superGlue`.

    Args:
        data_type (str): name of the task, supports ['nli', 'qa']
        data_path (str): path to the dataset file
    """
    assert data_type in task_to_cols, f"Task {data_type} not found."
    df = pd.read_csv(data_path, sep='\t')

    cols_map = task_to_cols[data_type]
    df = df[~df[list(cols_map.keys())[0]].isna()]

    cols_to_drop = [col for col in df.columns if col not in cols_map]
    df = df.drop(columns=cols_to_drop)

    df = df.rename(columns=cols_map)
    ds = datasets.Dataset.from_pandas(df)

    return ds

if __name__ == '__main__':
    task, data_path, = sys.argv[1:]

    ds = load_dataset(task, data_path)


