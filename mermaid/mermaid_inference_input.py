import spacy
from datasets import load_from_disk
from tqdm import tqdm

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

def generate_mermaid_input(sent, nlp):
    doc = nlp(sent)
    transfered_text = ' '.join([i.text if i.dep_ != 'ROOT' else '<V> '+i.text+' <V>' for i in doc])
    return transfered_text

def get_non_metaphor_to_produce_plain_text(task):
    """
        输出txt文件，每行一个句子。
    """
    ds = load_from_disk(f'glue_mermaid/{task}')
    col1, col2 = task_to_keys[task]
    f1 = open(f'glue_mermaid/mermaid_input/{task}_{col1}.input', 'w', encoding='utf-8')
    if col2 is not None:
        f2 = open(f'glue_mermaid/mermaid_input/{task}_{col2}.input', 'w', encoding='utf-8')
        
    for i in ds:
        if sum(i[f'{col1}_vua'])>0:
            f1.write(i[f'{col1}_mermaid']+'\n')
        if col2 is not None and sum(i[f'{col2}_vua'])>0:
            f2.write(i[f'{col2}_mermaid']+'\n')
    
    f1.close()
    if col2 is not None:
        f2.close()
    print(f'finished writing {task}.')

if __name__ == '__main__':
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'attribute_ruler', 'lemmatizer', 'tagger'])

    for task in tasks[6:]:
        ds = load_from_disk(f'glue_val/{task}')
        col1, col2 = task_to_keys[task]
        trans_col1 = []
        if col2 is not None:
            trans_col2 = []
        for i in ds:
            trans_col1.append(generate_mermaid_input(i[col1], nlp))
            if col2 is not None:
                trans_col2.append(generate_mermaid_input(i[col2], nlp))
        
        ds = ds.add_column(f'{col1}_mermaid', trans_col1)
        if col2 is not None:
            ds = ds.add_column(f'{col2}_mermaid', trans_col2)
        ds.save_to_disk(f'glue_mermaid/{task}')
        print(f'Finished {task}!')

                