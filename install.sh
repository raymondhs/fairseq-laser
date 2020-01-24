if [ -z ${FAIRSEQ+x} ] ; then
  echo "Please set the environment variable 'FAIRSEQ'"
  exit
fi

cd laser
cp add_decoder_lang_dataset.py multi_corpus_sampled_with_eval_key_dataset.py $FAIRSEQ/fairseq/data
cp laser.py $FAIRSEQ/fairseq/models
cp translation_laser.py $FAIRSEQ/fairseq/tasks
cd ..
