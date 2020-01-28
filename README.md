# fairseq-laser

Implementing LASER architecture using fairseq library based on my understanding of the original papers by Artexte and Schwenk (2018, [1] and [2]).

## Requirements

* Python version >= 3.6
* [PyTorch](https://pytorch.org/) (tested on version 1.3.1)
* [fairseq](https://github.com/pytorch/fairseq) (tested on version 0.9.0)
* [Faiss](https://github.com/facebookresearch/faiss), for bitext mining

## Training

This example shows how to train a LASER model on 4 languages from Europarl v7 (English/French/Spanish/German) with a similar architecture in [1].

```bash
# Download and preprocess the data
bash prepare-europarl.sh

# Binarize datasets for each language pair
bpe=europarl_en_de_es_fr/bpe.40k
data_bin=data-bin/europarl.de_en_es_fr.bpe40k
for lang_pair in de-en de-es de-fr en-es en-fr es-fr; do
    src=`echo $lang_pair | cut -d'-' -f1`
    tgt=`echo $lang_pair | cut -d'-' -f2`
    rm $data_bin/dict.$src.txt $data_bin/dict.$tgt.txt
    fairseq-preprocess --source-lang $src --target-lang $tgt \
        --trainpref $bpe/train.$src-$tgt \
        --joined-dictionary --tgtdict $bpe/vocab \
        --destdir $data_bin \
        --workers 20
done

# Train a LASER model. To speed up, we only use 2 target languages
# (English and Spanish) and train for 10 epochs.
checkpoint=checkpoints/laser_lstm
mkdir -p $checkpoint
fairseq-train $data_bin \
  --max-epoch 10 \
  --ddp-backend=no_c10d \
  --task translation_laser --arch laser \
  --lang-pairs de-en,de-es,en-es,es-en,fr-en,fr-es \
  --optimizer adam --adam-betas '(0.9, 0.98)' \
  --lr 0.001 --criterion cross_entropy \
  --dropout 0.1 --save-dir $checkpoint \
  --max-tokens 12000 --fp16 \
  --valid-subset train --disable-validation \
  --no-progress-bar --log-interval 1000 \
  --user-dir laser/
```

## Bitext mining

Here are some results on running the above model on [BUCC 2018 shared task data](https://comparable.limsi.fr/bucc2017/cgi-bin/download-data-2018.cgi) (see `bucc.sh`). The scores are on training set since the gold standard for the test set is not released (refer to Table 2 of [1] for comparison to a similar model).

| Languages | Threshold | Precision | Recall | F1 score |
|-----------|-----------|-----------|--------|---------|
|   fr-en   |  1.102786 |   91.63   |  91.37 |  91.50  |
|   de-en   |  1.095823 |   95.12   |  94.57 |  94.84  |

## References

[1] Mikel Artetxe and Holger Schwenk, [*Margin-based Parallel Corpus Mining with Multilingual Sentence Embeddings*](https://arxiv.org/abs/1811.01136) arXiv, Nov 3 2018.

[2] Mikel Artetxe and Holger Schwenk, [*Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond*](https://arxiv.org/abs/1812.10464) arXiv, 26 Dec 2018.
