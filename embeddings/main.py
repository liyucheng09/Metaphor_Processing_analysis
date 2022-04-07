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
import sys
import torch

class SenseEmbedding(BertWhitening):

    def __init__(self, model_path, **kwargs):
        super(SenseEmbedding, self).__init__(SentenceEmbeddingModel, model_path, **kwargs)    

    def preprocess_context(self, contexts: list[Context]):
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

    def get_embeddings(self, contexts: list[Context], whitening=False):
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
    words = [ 'help', 'look', 'bank']
    # words = [ 'bank.n' ]
    pool = 'idx-last-four-average'
    plot_types = ['tSNE', 'PCA']
    output_data_point_path = 'embeddings/datapoints'

    word2sentence = word2sentence('semcor')
    model = SenseEmbedding('roberta-base', add_prefix_space = True, pool = pool, max_length=100)
    # model = SenseEmbedding('bert-large-uncased', pool = pool, max_length=256)
    for word in words:
        contexts = word2sentence(word, minimum=2)
        vecs = model.get_embeddings(contexts)
        for plot_type in plot_types:
            X = plotDimensionReduction(vecs, [con.gloss for con in contexts], \
                figure_name=f'embeddings/imgs/{word}_{plot_type}_{pool}.png', plot_type=plot_type, \
                legend_loc=9, bbox_to_anchor=(0.5, -0.1))
            if output_data_point_path is not None:
                path = f'{output_data_point_path}/{word}_{plot_type}_{pool}.csv'
                f = open(path, 'w', encoding='utf-8')
                for point, context in zip(X, contexts):
                    f.write(f'{point[0]}\t{point[1]}\t{context.gloss}\t{" ".join([token.word if index != context.index else "[" + token.word +"]" for index, token in enumerate(context.tokens)])}\n')
                f.close()
                print(f'Saved to {path}!')