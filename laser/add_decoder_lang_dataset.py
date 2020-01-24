from fairseq.data import FairseqDataset, data_utils


class AddDecoderLangDataset(FairseqDataset):

    def __init__(self, dataset, decoder_lang):
        self.dataset = dataset
        self.decoder_lang = decoder_lang

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def collater(self, samples):

        if len(samples) == 0:
            return {}

        batch = self.dataset.collater(samples)
        batch['net_input']['decoder_lang'] = self.decoder_lang
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
