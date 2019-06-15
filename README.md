# fairseq-laser

Implementing LASER architecture using fairseq library.

## Training

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
