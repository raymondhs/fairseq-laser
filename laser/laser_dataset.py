import numpy as np
import torch

from fairseq.data import FairseqDataset, data_utils


class LaserDataset(FairseqDataset):

    def __init__(self, dataset, trg_langid):
        self.dataset = dataset
        self.trg_langid = trg_langid

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):

        if len(samples) == 0:
            return {}

        def merge(key, left_pad, move_eos_to_beginning=False):
            return data_utils.collate_tokens(
                [s[key] for s in samples],
                self.dataset.src_dict.pad(),
                self.dataset.src_dict.eos(), left_pad, move_eos_to_beginning,
            )

        batch = self.dataset.collater(samples)
        trg_segments = None
        if samples[0].get('target', None) is not None:
            # add target language id
            for s in samples:
                trg_length = s['target'].numel()
                trg_segments = np.ones(trg_length) * self.trg_langid
                s['trg_segments'] = torch.LongTensor(trg_segments)
            batch['net_input']['trg_segments'] = merge('trg_segments', left_pad=self.dataset.left_pad_target)
        return batch

    def num_tokens(self, index):
        return self.dataset.num_tokens(index)

    def size(self, index):
        return self.dataset.size(index)

    def ordered_indices(self):
        return self.dataset.ordered_indices()

    @property
    def supports_prefetch(self):
        return getattr(self.dataset, 'supports_prefetch', False)

    def prefetch(self, indices):
        return self.dataset.prefetch(indices)
