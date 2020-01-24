# fairseq-laser

Implementing LASER architecture using fairseq library. Refer to the `examples` directory for running experiments on similarity search and bitext mining similar to Artexte and Schwenk (2018).

## Install

This code has been tested on PyTorch 1.3.1 and fairseq 0.9.0. To install:

* Make sure fairseq installation is editable (`pip install --editable /path/to/fairseq`).
* Set the environment variable 'FAIRSEQ' to fairseq root (`export FAIRSEQ=/path/to/fairseq`).
* Run `bash ./install.sh`. This will copy the files in `laser` to appropriate directories in `$FAIRSEQ`.

## Training

First, pre-process your data (e.g., using this [example script](https://github.com/pytorch/fairseq/blob/master/examples/translation/prepare-iwslt17-multilingual.sh)). Example usage for training:

```
$ mkdir -p checkpoints/laser
$ CUDA_VISIBLE_DEVICES=0 fairseq-train data-bin/iwslt17.de_fr.en.bpe16k/ \
    --max-epoch 50 \
    --ddp-backend=no_c10d \
    --task translation_laser --lang-pairs de-en,fr-en \
    --arch laser --encoder-num-layers 5 \
    --encoder-embed-dim 320 --decoder-embed-dim 320 \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr 0.001 --criterion cross_entropy \
    --save-dir checkpoints/laser \
    --max-tokens 4000 \
    --update-freq 8
```

## Generate Embeddings

```
$ SRC=de
$ sacrebleu --test-set iwslt17 --language-pair ${SRC}-en --echo src > iwslt17.test.${SRC}-en.${SRC}
$ cat iwslt17.test.${SRC}-en.${SRC} | python embed.py data-bin/iwslt17.de_fr.en.bpe16k/ \
  --task translation_laser --lang-pairs de-en,fr-en \
  --source-lang ${SRC} --target-lang en \
  --path checkpoints/laser/checkpoint_best.pt \
  --buffer-size 2000 --batch-size 128 \
  --output-file iwslt17.test.${SRC}-en.${SRC}.enc
```

### Output format

Same as the original LASER format, the embeddings are stored in float32 matrices in raw binary format.
They can be read in Python by:
```
import numpy as np
dim = 1024
X = np.fromfile("iwslt17.test.de-en.de.enc", dtype=np.float32, count=-1)                                                                          
X.resize(X.shape[0] // dim, dim)                                                                                                 
```
X is a N x 1024 matrix where N is the number of lines in the text file.

## References

Mikel Artetxe and Holger Schwenk,
    [*Margin-based Parallel Corpus Mining with Multilingual Sentence Embeddings*]
    (https://arxiv.org/abs/1811.01136)
    arXiv, Nov 3 2018.

Mikel Artetxe and Holger Schwenk,
    [*Massively Multilingual Sentence Embeddings for Zero-Shot Cross-Lingual Transfer and Beyond*](https://arxiv.org/abs/1812.10464)
    arXiv, 26 Dec 2018.
