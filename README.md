# fairseq-laser

Implementing LASER architecture using fairseq library.

## Training

First, pre-process your data (e.g., using SentencePiece) [example script](https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-iwslt17-multilingual.sh). Example usage for training:

```
$ mkdir -p checkpoints/laser_lstm
$ fairseq-train data-bin/iwslt17.de_fr.en.bpe16k/ \
  --max-epoch 17 \
  --ddp-backend=no_c10d \
  --task translation_laser --lang-pairs de-en,fr-en \
  --arch laser_lstm_artetxe \
  --encoder-num-layers 5 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr 0.001 --lr-scheduler fixed \
  --weight-decay 0.0 --criterion cross_entropy \
  --save-dir checkpoints/laser_lstm \
  --max-tokens 3584 \
  --update-freq 8 \
  --no-progress-bar --log-interval 50 \
  --user-dir $PWD/laser/
```

## Generate Embeddings

```
$ SRC=de
$ cat iwslt17.test.${SRC}-en.${SRC}.bpe | python embed.py data-bin/iwslt17.de_fr.en.bpe16k/ \
  --task translation_laser --source-lang ${SRC} -- target-lang en \
  --path checkpoints/laser_lstm/checkpoint_best.pt \
  --buffer 2000 --batch-size 128 \
  --output-file iwslt17.test.${SRC}-en.${SRC}.enc
```

### Output format

Same as the original LASER format, the embeddings are stored in float32 matrices in raw binary format.
They can be read in Python by:
```
import numpy as np
dim = 1024
X = np.fromfile("iwslt17.test.${SRC}-en.${SRC}.enc", dtype=np.float32, count=-1)                                                                          
X.resize(X.shape[0] // dim, dim)                                                                                                 
```
X is a N x 1024 matrix where N is the number of lines in the text file.

## References

Mikel Artetxe and Holger Schwenk,
    [*Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond*](https://arxiv.org/abs/1812.10464)
    arXiv, 26 Dec 2018.
