import datasets
import pandas as pd
import math

def compute_bleu_score(x):
    if x['metaphor_symbol'] == 0 or x['literal_symbol'] == 0:
        return None
    metaphor_symbols = eval(x['metaphor_symbol'])
    literal_symbols = eval(x['literal_symbol'])

    metaphor_symbols = ' '.join(metaphor_symbols).split(' ')
    literal_symbols = ' '.join(literal_symbols).split(' ')

    literal_symbols = [[literal_symbols]]
    metaphor_symbols = [metaphor_symbols]
    bleu_score = bleu.compute(predictions=metaphor_symbols, references=literal_symbols, max_order=1)

    return bleu_score['bleu']

if __name__ == '__main__':

    df = pd.read_csv('Metaphor-Emotion-Data-Files/moh.conceptnet.tsv', sep='\t')
    df = df.fillna(0)
    bleu = datasets.load_metric('bleu')

    bleu_score = df.apply(compute_bleu_score, axis=1)
    df['bleu_score'] = bleu_score

    df.to_csv('Metaphor-Emotion-Data-Files/moh.conceptnet.bleu.tsv', index=False, sep='\t')