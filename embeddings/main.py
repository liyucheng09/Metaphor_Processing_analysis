import sys
sys.path.append('/user/HS502/yl02706/mpa')
from lyc.utils import (get_model,
                       get_tokenizer,
                       vector_l2_normlize,
                       get_vectors,
                       compute_kernel_bias,
                       save_kernel_and_bias)
from lyc.model import SentenceEmbeddingModel
from lyc.pipelines.bert_whitening import BertWhitening
from lyc.visualize import plotDimensionReduction
from util import word2sentence, sense, Token, Context
import pickle
import torch
from typing import List
import os

class SenseEmbedding(BertWhitening):

    def __init__(self, model_path, **kwargs):
        super(SenseEmbedding, self).__init__(SentenceEmbeddingModel, model_path, **kwargs)    

    def preprocess_context(self, contexts: List[Context]):
        sents = []
        idxs = []
        senses = []
        new_idxs = []

        for cont in contexts:
            idx = cont.index
            target_sense = cont.tokens[idx].sense
            sent = [t.word for t in cont.tokens]
            
            sents.append(sent)
            idxs.append(idx)
            senses.append(target_sense)
        
        output = self.tokenizer(sents, is_split_into_words = True, max_length = self.max_length, padding=True, truncation = True, return_tensors = 'pt')
        for i in range(len(sents)):
            new_idx = output.word_ids(i)
            new_idx = new_idx.index(idxs[i])
            new_idxs.append(new_idx)
        
        # new_idxs = torch.LongTensor(new_idxs)
        
        return output, new_idxs, senses

    def get_embeddings(self, contexts: List[Context], whitening=False):
        encoding, idxs, senses = self.preprocess_context(contexts)
        with torch.no_grad():
            vecs=get_vectors(self.model, encoding, idxs = idxs)
        vecs=vecs.cpu().numpy()

        if whitening:
            assert self.enable_whitening
            if self.dynamic_kernel:
                vecs=vector_l2_normlize(vecs)
                kernel, bias = compute_kernel_bias([vecs])
            else:
                kernel, bias = self.kernel, self.bias
            kernel=kernel[:, :self.n_components]
            vecs=transform_and_normalize(vecs, kernel, bias)
            return vecs
        vecs=vector_l2_normlize(vecs)
        return vecs
    
    def _computing_kernel_and_save(self, kernel_bias_path, corpus_for_kernel_computing):
        with open(corpus_for_kernel_computing, 'rb') as f:
            contexts = pickle.load(f)
        vecs=self.get_embeddings(contexts, whitening=False)
        kernel, bias = compute_kernel_bias([vecs])
        save_kernel_and_bias(kernel, bias, kernel_bias_path)


if __name__ == '__main__':
    cwd, = sys.argv[1:]

    index_path = os.path.join(cwd, 'embeddings/index')
    output_data_point_path = os.path.join(cwd, 'embeddings/datapoints')
    tokenizer = get_tokenizer('roberta-base', add_prefix_space=True)
    
    # words = [ 'act', 'admit', 'age', 'address', 'answer', 'ask', 'breathe', 'buy', 'consider', 'cook', 'distill', 'end', 'fire', 'head']
    words = [ 'bank.n', 'activate.v', 'lose.v', 'play.v', 'image.n']
    pool = 'idx-last'
    # pool = 'idx-last-four-average'
    plot_types = ['PCA']
    # model_paths = [f'/vol/research/lyc/mpa/senseCL/checkpoint/checkpoint-{i}' for i in range(100, 600, 100)]
    model_paths = ['checkpoints/senseCL/checkpoint-400']

    word2sentence = word2sentence('senseval3', index_path = index_path)
    # model = SenseEmbedding('bert-large-uncased', pool = pool, max_length=256)
    for model_path in model_paths:
        model_id = os.path.basename(model_path)
        model = SenseEmbedding(model_path=model_path, add_prefix_space = True, pool = pool, max_length=128, output_hidden_states = True)

        for word in words:
            contexts = word2sentence(word, minimum=2, max_length=128)
            if not contexts:
                print(f'{word} do not have enough contexts to visualize!')
                continue
            contexts.append(Context(tokens=[Token(word, '_', '_', f'blend_of_{word}')], index=0, gloss=f'blend_of_{word}'))
            vecs = model.get_embeddings(contexts)
            for plot_type in plot_types:
                X = plotDimensionReduction(vecs, [con.gloss for con in contexts], \
                    figure_name= os.path.join(cwd, f'embeddings/imgs/senseCL/{word}_{plot_type}_{pool}_{model_id}.png'), plot_type=plot_type, \
                    legend_loc=9, bbox_to_anchor=(0.5, -0.1))
                if output_data_point_path is not None:
                    path = f'{output_data_point_path}/{word}_{plot_type}_{pool}_{model_id}.csv'
                    f = open(path, 'w', encoding='utf-8')
                    for point, context in zip(X, contexts):
                        f.write(f'{point[0]}\t{point[1]}\t{context.gloss}\t{" ".join([token.word if index != context.index else "[" + token.word +"]" for index, token in enumerate(context.tokens)])}\n')
                    f.close()
                    print(f'Saved to {path}!')
