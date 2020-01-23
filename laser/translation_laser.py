from collections import OrderedDict
import logging
import os

from fairseq import options, utils
from fairseq.data import (
    Dictionary,
    LanguagePairDataset,
)
from fairseq.tasks import FairseqTask, register_task
from fairseq.tasks.multilingual_translation import MultilingualTranslationTask, load_langpair_dataset

from fairseq.data.add_decoder_lang_dataset import AddDecoderLangDataset
from fairseq.data.multi_corpus_sampled_with_eval_key_dataset import MultiCorpusSampledWithEvalKeyDataset


@register_task('translation_laser')
class TranslationLaserTask(FairseqTask):

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        MultilingualTranslationTask.add_args(parser)
        # fmt: on

    def __init__(self, args, dicts, training):
        super().__init__(args)
        self.dicts = dicts
        self.training = training
        if training:
            self.lang_pairs = args.lang_pairs
            args.source_lang, args.target_lang = args.lang_pairs[0].split('-')
        else:
            self.lang_pairs = ['{}-{}'.format(args.source_lang, args.target_lang)]
        self.langs = list(dicts.keys())

    @classmethod
    def setup_task(cls, args, **kwargs):
        dicts, training = MultilingualTranslationTask.prepare(args, **kwargs)
        return cls(args, dicts, training)

    def load_dataset(self, split, epoch=0, **kwargs):
        """Load a dataset split."""

        paths = self.args.data.split(':')
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]

        def language_pair_dataset(lang_pair):
            src, tgt = lang_pair.split('-')
            langpair_dataset = load_langpair_dataset(
                data_path, split, src, self.dicts[src], tgt, self.dicts[tgt],
                combine=True, dataset_impl=self.args.dataset_impl,
                upsample_primary=self.args.upsample_primary,
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions,
            )
            return AddDecoderLangDataset(langpair_dataset, tgt)

        self.datasets[split] = MultiCorpusSampledWithEvalKeyDataset(
            OrderedDict([
                (lang_pair, language_pair_dataset(lang_pair))
                for lang_pair in self.lang_pairs
            ]),
            eval_key=None if self.training else "%s-%s" % (self.args.source_lang, self.args.target_lang),
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths):
        lang_pair = "%s-%s" % (self.args.source_lang, self.args.target_lang)
        return MultiCorpusSampledWithEvalKeyDataset(
            OrderedDict([(
                lang_pair,
                AddDecoderLangDataset(
                    LanguagePairDataset(
                        src_tokens, src_lengths,
                        self.source_dictionary
                    ),
                    self.args.target_lang,
                ),
            )]),
            eval_key=lang_pair,
        )

    def build_model(self, args):
        # Check if task args are consistant with model args
        if len(set(self.args.lang_pairs).symmetric_difference(args.lang_pairs)) != 0:
            raise ValueError('--lang-pairs should include all the language pairs {}.'.format(args.lang_pairs))

        from fairseq import models
        model = models.build_model(args, self)
        from fairseq.models.laser import LaserModel
        if not isinstance(model, LaserModel):
            raise ValueError('TranslationLaserTask requires a LaserModel architecture')
        return model

    @property
    def source_dictionary(self):
        return self.dicts[self.args.source_lang]

    @property
    def target_dictionary(self):
        return self.dicts[self.args.target_lang]
