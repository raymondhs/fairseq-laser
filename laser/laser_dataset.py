import numpy as np
import torch

from fairseq.data import FairseqDataset, data_utils


class LaserDataset(FairseqDataset):

    def __init__(self, dataset, tgt_langid):
        self.dataset = dataset
        self.tgt_langid = tgt_langid

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):

        if len(samples) == 0:
            return {}

        def merge(key, left_pad):
            return data_utils.collate_tokens(
                [s[key] for s in samples], 0, left_pad=left_pad,
            )

        batch = self.dataset.collater(samples)
        # add target language id
        for s in samples:
            tgt_length = s['target'].numel()
            tgt_segments = np.ones(tgt_length) * self.tgt_langid
            s['tgt_segments'] = torch.LongTensor(tgt_segments)
        batch['net_input']['tgt_segments'] = merge('tgt_segments', left_pad=self.dataset.left_pad_target)
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
