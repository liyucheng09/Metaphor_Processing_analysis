import sys
sys.path.append('/user/HS502/yl02706/mpa')
import datasets
import pandas as pd
from lyc.utils import get_model, get_tokenizer
from transformers import AutoModelForSequenceClassification, Trainer, RobertaForSequenceClassification
from transformers import default_data_collator
from lyc.train import get_base_hf_args
from lyc.eval import write_predict_to_file
import os

task_to_cols = {
    'nli': {'Hypothesis' : 'sentence1', 'Claim': 'sentence2', 'Answer': 'label'},
    'qa': {'Question': 'question', 'context': 'passage', 'Answer': 'label'},
    'qa_non': {'Question': 'question', 'context': 'passage', 'Answer': 'label'}
}
task_to_keys = {
    'nli': ("sentence1", "sentence2"),
    'qa': ('question', 'passage'),
    'qa_non': ('question', 'passage')
}

label2id = {'F':0, 'T':1}

def load_dataset(data_type, data_path):
    """load metaphor dataset with downstream task annotations.
    Task `nli` should be test via the `MRPC` model.
    Task `qa` is a binary question answering problem with context. Should be test via `BoolQ` model from `superGlue`.

    Args:
        data_type (str): name of the task, supports ['nli', 'qa']
        data_path (str): path to the dataset file
    """
    def remove_brackets(x):
        tokens = x.split()
        tokens = [token[1:-1] if token.startswith('[') and token.endswith(']') else token for token in tokens]
        return ' '.join(tokens)

    assert data_type in task_to_cols, f"Task {data_type} not found."
    df = pd.read_csv(data_path, sep='\t')

    cols_map = task_to_cols[data_type]
    df = df[~df[list(cols_map.keys())[0]].isna()]

    cols_to_drop = [col for col in df.columns if col not in cols_map]
    df = df.drop(columns=cols_to_drop)
    if 'context' in df.columns:
        print('has context')
        df['context'] = df['context'].apply(remove_brackets)

    df = df.rename(columns=cols_map)
    ds = datasets.Dataset.from_pandas(df)

    return ds

if __name__ == '__main__':
    task, data_path, model_name, output_path = sys.argv[1:]
    max_length = 256
    output_path = os.path.join(output_path, task)
    output_file = os.path.join(output_path, 'predictions.tsv')

    ds = load_dataset(task, data_path)

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = get_tokenizer(model_name)
    args = get_base_hf_args(
        output_dir = output_path,
    )
    
    col1, col2 = task_to_keys[task]

    def preprocess(examples):
        out = tokenizer(examples[col1], examples[col2], padding=True, max_length = max_length, truncation = True)
        out['label'] = [label2id[l] for l in examples['label']]
        return out
    
    ds = ds.map(preprocess, batched=True)

    trainer = Trainer(
        model = model,
        args = args,
        data_collator=default_data_collator
    )

    prediction = trainer.predict(ds)
    write_predict_to_file(prediction, out_file=output_file)

