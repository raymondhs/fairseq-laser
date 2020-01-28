#!/bin/bash
set -e

# bash script from LASER repo to mine for bitexts in the BUCC corpus
# (https://github.com/facebookresearch/LASER/blob/master/tasks/bucc/bucc.sh)
# modified to use fairseq to generate sentence embeddings


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
if [ ! -d LASER ] ; then
  echo 'Cloning LASER github repository...'
  git clone https://github.com/facebookresearch/LASER.git
fi
export LASER=$PWD/LASER

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
NORM_PUNC=$SCRIPTS/tokenizer/normalize-punctuation.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
REM_NON_PRINT_CHAR=$SCRIPTS/tokenizer/remove-non-printing-char.perl

# general config
bucc="bucc2018"
data="."
xdir=${data}/downloaded	# tar files as distrubuted by the BUCC evaluation
ddir=${data}/${bucc}	# raw texts of BUCC
edir=${data}/embed	# normalized texts and embeddings
langs=("fr" "de")
ltrg="en"		# English is always the 2nd language

# encoder
data_bin=data-bin/europarl.de_en_es_fr.bpe40k/
bpe=europarl_en_de_es_fr/bpe.40k
checkpoint=checkpoints/laser_lstm/checkpoint_last.pt

# delete all generated files to re-run
rerun=true
if [ $rerun = true ] ; then
  rm -f $edir/*.enc.* $edir/*.candidates.tsv bucc2018.*.train.log
fi

###################################################################
#
# Extract files with labels and texts from the BUCC corpus
#
###################################################################

GetData () {
  fn1=$1; fn2=$2; lang=$3
  outf="${edir}/${bucc}.${lang}-${ltrg}.${fn2}"
  for ll  in ${ltrg} ${lang} ; do
    inf="${ddir}/${fn1}.${ll}"
    if [ ! -f ${outf}.txt.${ll} ] ; then
      echo " - extract files ${outf} in ${ll}"
      cat ${inf} | cut -f1 > ${outf}.id.${ll}
      cat ${inf} | cut -f2 > ${outf}.txt.${ll}
    fi
  done
}

ExtractBUCC () {
  slang=$1
  tlang=${ltrg}

  pushd ${data} > /dev/null
  if [ ! -d ${ddir}/${slang}-${tlang} ] ; then
    for tf in ${xdir}/${bucc}-${slang}-${tlang}.*.tar.bz2 ; do
      echo " - extract from tar `basename ${tf}`"
      tar jxf $tf
    done
  fi

  GetData "${slang}-${tlang}/${slang}-${tlang}.sample" "dev" ${slang}
  GetData "${slang}-${tlang}/${slang}-${tlang}.training" "train" ${slang}
  GetData "${slang}-${tlang}/${slang}-${tlang}.test" "test" ${slang}
  popd > /dev/null
}


###################################################################
#
# Tokenize and Embed
#
###################################################################

Embed () {
  ll=$2
  txt="$1.txt.${ll}"
  enc="$1.enc.${ll}"
  tl="en"
  if [ $ll = "en" ]; then tl="es" ; fi
  if [ ! -s ${enc} ] ; then
    cat ${txt} | \
      perl $REM_NON_PRINT_CHAR | \
      perl $NORM_PUNC $l | \
      perl $TOKENIZER -threads 20 -l $ll -q -no-escape | \
      perl $LC | \
      fastBPE/fast applybpe_stream $bpe/codes $bpe/vocab | \
      python3 embed.py $data_bin \
        --task translation_laser \
        --lang-pairs de-en,de-es,en-es,es-en,fr-en,fr-es \
        --source-lang $ll --target-lang $tl \
        --path $checkpoint \
        --buffer-size 2000 --batch-size 128 \
        --output-file ${enc} \
        --user-dir laser/
  fi
}


###################################################################
#
# Mine for bitexts
#
###################################################################

Mine () {
  bn=$1
  l1=$2
  l2=$3
  cand="${bn}.candidates.tsv"
  if [ ! -s ${cand} ] ; then
    python3 ${LASER}/source/mine_bitexts.py \
       ${bn}.txt.${l1} ${bn}.txt.${l2} \
       --src-lang ${l1} --trg-lang ${l2} \
       --src-embeddings ${bn}.enc.${l1} --trg-embeddings ${bn}.enc.${l2} \
       --unify --mode mine --retrieval max --margin ratio -k 4  \
       --output ${cand} \
       --verbose --gpu
  fi
}


###################################################################
#
# Main loop
#
###################################################################

echo -e "\nProcessing BUCC data in ${data}"

# create output directories
for d in ${ddir} ${edir} ; do
  mkdir -p ${d}
done

for lsrc in ${langs[@]} ; do
  ExtractBUCC ${lsrc}

  # Tokenize and embed train 
  bname="${bucc}.${lsrc}-${ltrg}"
  part="${bname}.train"
  Embed ${edir}/${part} ${lsrc} ${encoder} ${bpe_codes}
  Embed ${edir}/${part} ${ltrg} ${encoder} ${bpe_codes}

  # mine for texts in train
  Mine ${edir}/${part} ${lsrc} ${ltrg}

  # optimize threshold on BUCC training data and provided gold alignments
  if [ ! -s ${part}.log ] ; then
    python3 ${LASER}/tasks/bucc/bucc.py \
      --src-lang ${lsrc} --trg-lang ${ltrg} \
      --bucc-texts ${edir}/${part}.txt \
      --bucc-ids ${edir}/${part}.id \
      --candidates ${edir}/${part}.candidates.tsv \
      --gold ${ddir}/${lsrc}-${ltrg}/${lsrc}-${ltrg}.training.gold \
      --verbose \
      | tee ${part}.log
  fi
done
