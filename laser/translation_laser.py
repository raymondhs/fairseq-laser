from fairseq.tasks import register_task
from fairseq.tasks.multilingual_translation import MultilingualTranslationTask, load_langpair_dataset

from .laser_dataset import LaserDataset

@register_task('translation_laser')
class TranslationLaserTask(MultilingualTranslationTask):

    def __init__(self, args, dicts, training):
        super().__init__(args, dicts, training)
        trg_lang2id = dict()
        for lang_pair in args.lang_pairs:
            _, trg_lang = lang_pair.split('-')
            trg_lang2id.setdefault(trg_lang, len(trg_lang2id)+1)
        self.trg_lang2id = trg_lang2id

    def alter_dataset_langtok(self, lang_pair_dataset,
                              src_eos=None, src_lang=None, tgt_eos=None, tgt_lang=None):
        dataset = super().alter_dataset_langtok(lang_pair_dataset, src_eos, src_lang, tgt_eos, tgt_lang)
        return LaserDataset(dataset, self.trg_lang2id[tgt_lang])
