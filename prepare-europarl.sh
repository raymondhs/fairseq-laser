#!/bin/bash

if [ ! -d mosesdecoder ] ; then
  echo 'Cloning Moses github repository (for tokenization scripts)...'
  git clone https://github.com/moses-smt/mosesdecoder.git
fi
if [ ! -d fastBPE ] ; then
  git clone https://github.com/glample/fastBPE.git
  pushd fastBPE
  g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
  popd
fi

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl

prep=europarl_en_de_es_fr
tmp=$prep/tmp
orig=$prep/downloaded
codes=40000
bpe=$prep/bpe.40k

mkdir -p $orig $tmp $bpe

urlpref=http://opus.nlpl.eu/download.php?f=Europarl/v8/moses
mkdir -p $orig
for f in en-es.txt.zip de-en.txt.zip de-es.txt.zip de-fr.txt.zip en-fr.txt.zip es-fr.txt.zip de-fr.txt.zip; do
  if [ ! -f $orig/$f ] ; then
    wget $urlpref/$f -O $orig/$f
    rm $orig/{README,LICENSE}
    unzip $orig/$f -d $orig
  fi
done

echo "pre-processing train data..."
for lang_pair in de-en de-es de-fr en-es en-fr es-fr ; do
    src=`echo $lang_pair | cut -d'-' -f1`
    tgt=`echo $lang_pair | cut -d'-' -f2`
    lang=$src-$tgt
    for l in $src $tgt; do
        rm  -rf $tmp/train.tags.$lang.tok.$l
        for f in Europarl ; do
            cat $orig/$f.$lang.$l | \
                perl $REM_NON_PRINT_CHAR | \
                perl $NORM_PUNC $l | \
                perl $TOKENIZER -threads 20 -l $l -q -no-escape | \
                perl $LC >> $tmp/train.tags.$lang.tok.$l
        done
    done
done

rm -f $tmp/train.all
# apply length filtering before BPE
for lang_pair in de-en de-es de-fr en-es en-fr es-fr ; do
    src=`echo $lang_pair | cut -d'-' -f1`
    tgt=`echo $lang_pair | cut -d'-' -f2`
    lang=$src-$tgt
    perl $CLEAN -ratio 1.5 $tmp/train.tags.$lang.tok $src $tgt $tmp/train.$lang 1 100
    cat $tmp/train.$lang.{$src,$tgt} >> $tmp/train.all
done


#BPE
fastBPE/fast learnbpe $codes $tmp/train.all > $bpe/codes
for lang_pair in de-en de-es de-fr en-es en-fr es-fr ; do
    src=`echo $lang_pair | cut -d'-' -f1`
    tgt=`echo $lang_pair | cut -d'-' -f2`
    lang=$src-$tgt
    for l in $src $tgt; do
        fastBPE/fast applybpe $bpe/train.$lang.$l $tmp/train.$lang.$l $bpe/codes
    done
done

cat $bpe/train.*.* | fastBPE/fast getvocab - > $bpe/vocab
