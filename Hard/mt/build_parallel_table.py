from lyc.utils import DeepLTranslator
import pandas as pd

def remove_brackets(x):
    return x.replace('[', '').replace(']', '')

if __name__ == '__main__':
    translator = DeepLTranslator()

    df = pd.read_csv('Hard/mt/literal.tsv', sep='\t')
    # df = df.head()
    debrackets_sents = df['sent'].apply(remove_brackets)
    sents_to_translate = debrackets_sents.to_list()

    results = translator.translate(sents_to_translate, target_lang='ZH', source_lang='EN')
    translated_results = [i.text for i in results]

    df['translated'] = translated_results
    df.to_csv('Hard/mt/literal_translated.tsv', index=False, sep='\t')