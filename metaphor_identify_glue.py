tasks = ['cola', 'sst2', 'mrpc', 'stsb', 'qqp', 'mnli', 'qnli', 'rte', 'wnli']
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}
from lyc.utils import get_model, get_tokenizer
import os
from datasets import load_dataset
from transformers import RobertaForTokenClassification, DataCollatorForTokenClassification, Trainer
from lyc.train import get_base_hf_args
import sys
import numpy as np
import datasets

def data_preprocess(examples, col):
    result = tokenizer(examples[col])
    return result

def get_token_label_parallel(tokens, attention_mask, labels):
    true_label = []
    true_token = []
    for token, att_mask, label in zip(tokens, attention_mask, labels):
        ll = []
        tl = []
        for t,a,l in zip(token, att_mask, label):
            ll.append(l)
            tl.append(tokenizer.convert_ids_to_tokens(t))
        assert len(ll) == len(tl)
        true_label.append(ll)
        true_token.append(tl)
    return true_token, true_label

if __name__ == '__main__':
    model_name, task_name, save_folder, model_type = sys.argv[1:]
    # save_folder = '/vol/research/nlg/mpa/'
    # save_folder = './'

    output_dir = os.path.join(save_folder, f'{model_type}/{task_name}')
    logging_dir = os.path.join(save_folder, 'logs/')
    # prediction_output_file = os.path.join(output_dir, 'output_labels.csv')

    tokenizer = get_tokenizer(model_name)

    if task_name != 'mnli':
        ds = load_dataset('glue', task_name)['validation']
    else:
        ds1 = load_dataset('glue', task_name)['validation_matched']
        ds2 = load_dataset('glue', task_name)['validation_mismatched']
        ds = datasets.concatenate_datasets([ds1, ds2])
        del ds1, ds2

    col1, col2 = task_to_keys[task_name]
    ds1 = ds.map(data_preprocess, fn_kwargs={'col':col1}, remove_columns=ds.column_names)
    if col2 is not None:
        ds2 = ds.map(data_preprocess, fn_kwargs={'col':col2}, remove_columns=ds.column_names)
    
    data_collator = DataCollatorForTokenClassification(tokenizer, max_length=128)
    model = get_model(RobertaForTokenClassification, model_name, num_labels=2)

    args = get_base_hf_args(output_dir = output_dir)
    trainer = Trainer(
        model=model,
        args=args,
        # train_dataset=ds['train'],
        # eval_dataset=ds['test'],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    pred_out1 = trainer.predict(ds1)
    predictions_1 = np.argmax(pred_out1.predictions, axis=-1)

    true_token_1, true_label_1 = get_token_label_parallel(ds1['input_ids'], ds1['attention_mask'], predictions_1)    

    ds = ds.add_column(col1+f'_{model_type}', true_label_1)
    ds = ds.add_column(col1+f'_{model_type}_token', true_token_1)

    if col2 is not None:
        pred_out2 = trainer.predict(ds2)
        predictions_2 = np.argmax(pred_out2.predictions, axis=-1)

        true_token_2, true_label_2 = get_token_label_parallel(ds2['input_ids'], ds2['attention_mask'], predictions_2)    

        ds = ds.add_column(col2+f'_{model_type}', true_label_2)
        ds = ds.add_column(col2+f'_{model_type}_token', true_token_2)

    ds.save_to_disk(output_dir)