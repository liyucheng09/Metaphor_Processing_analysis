$HOME/.conda/envs/lyc/bin/python \
$HOME/mpa/mermaid/fairseq/inference.py \
--infile $HOME/mpa/mermaid/data/val.source \
--outfile $HOME/mpa/mermaid/metaphor_generated.val \
--apply_disc \
--scorers $HOME/mpa/mermaid/fairseq/WP_scorers.tsv \
--bart /vol/research/nlg/mermaid/checkpoint-metaphor \
--bart_data_path $HOME/mpa/mermaid/fairseq/metaphor


bart = BARTModel.from_pretrained(
    '/vol/research/nlg/mermaid/checkpoint-metaphor',
    checkpoint_file='checkpoint_best.pt',
    data_name_or_path='/user/HS502/yl02706/mpa/mermaid/fairseq/metaphor'
)

bart.cuda()