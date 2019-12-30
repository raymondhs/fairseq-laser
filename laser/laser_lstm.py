from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.models import (
    FairseqMultiModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.models.lstm import (
    Embedding,
    Linear,
    LSTM,
    LSTMEncoder,
)

from .translation_laser import TranslationLaserTask


class LaserLSTMEncoder(LSTMEncoder):

    def forward(self, src_tokens, src_lengths, **kwargs):
        encoder_out = super().forward(src_tokens, src_lengths)

        encoder_padding_mask = encoder_out['encoder_padding_mask']
        x, final_hiddens, final_cells = encoder_out['encoder_out']

        if encoder_padding_mask is not None:
            # Set padded outputs to -inf so they are not selected by max-pooling
            padding_mask = encoder_padding_mask.unsqueeze(-1)
            if padding_mask.any():
                x = x.float().masked_fill_(padding_mask, float('-inf')).type_as(x)

        # Build the sentence embedding by max-pooling over the encoder outputs
        sentemb = x.max(dim=0)[0]

        return {
            'sentemb': sentemb,
            'encoder_out': (x, final_hiddens, final_cells),
            'encoder_padding_mask': encoder_padding_mask
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        reordered = super().reorder_encoder_out(encoder_out, new_order)
        sentemb = encoder_out['sentemb']
        reordered['sentemb'] = sentemb.index_select(0, new_order)
        return reordered


class LaserLSTMDecoder(FairseqIncrementalDecoder):
    ''' LASER-style LSTM decoder '''
    def __init__(
        self, dictionary, encoder_output_units=512, embed_dim=512,
        hidden_size=512, num_langs=0, lang_embed_dim=32,
        dropout_in=0.1, dropout_out=0.1, pretrained_embed=None,
    ):
        super().__init__(dictionary)
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.hidden_size = hidden_size

        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.encoder_output_units = encoder_output_units
        input_size = encoder_output_units + embed_dim
        self.embed_lang = None
        if num_langs > 0:
            self.embed_lang = Embedding(
                num_embeddings=num_langs+1,
                embedding_dim=lang_embed_dim,
                padding_idx=0,
            )
            # include language embedding size
            input_size += lang_embed_dim

        # single layer decoder
        self.lstm = LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            dropout=0.,
            bidirectional=False,
        )

        # for projecting max pool sentemb from encoder
        self.sentemb_hidden_proj = Linear(encoder_output_units, hidden_size)
        self.sentemb_cell_proj = Linear(encoder_output_units, hidden_size)
        self.fc_out = Linear(hidden_size, num_embeddings)

    def forward(self, prev_output_tokens, encoder_out, tgt_segments=None, incremental_state=None):
        sentemb = encoder_out['sentemb']

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
        bsz, seqlen = prev_output_tokens.size()

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
        # optional language ID embedding
        if self.embed_lang is not None:
            if tgt_segments is None:
                raise ValueError('Current model with lang embeddings does not support inference.')
            x_lang = self.embed_lang(tgt_segments)
            x = torch.cat([x, x_lang], dim=2)
        x = F.dropout(x, p=self.dropout_in, training=self.training)
        x = torch.cat(
            [x, sentemb.unsqueeze(1).expand(bsz, seqlen, -1)],
            dim=2,
        )

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # initialize previous states (or get from cache during incremental generation)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is None:
            prev_hiddens = self.sentemb_hidden_proj(sentemb).unsqueeze(0)
            prev_cells = self.sentemb_cell_proj(sentemb).unsqueeze(0)
            cached_state = (prev_hiddens, prev_cells)

        # Run one step of our LSTM.
        output, latest_state = self.lstm(x, cached_state)
        output = F.dropout(output, p=self.dropout_out, training=self.training)

        # Update the cache with the latest hidden and cell states.
        utils.set_incremental_state(
            self, incremental_state, 'cached_state', latest_state,
        )

        # T x B x C -> B x T x C
        x = output.transpose(1, 0)
        x = self.fc_out(x)
        return x, None

    def reorder_incremental_state(self, incremental_state, new_order):
        super().reorder_incremental_state(incremental_state, new_order)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is None:
            return

        new_state = (
            cached_state[0].index_select(1, new_order),
            cached_state[1].index_select(1, new_order),
        )

        utils.set_incremental_state(self, incremental_state, 'cached_state', new_state)


@register_model('laser_lstm')
class LaserLSTMModel(FairseqMultiModel):

    def __init__(self, encoders, decoders):
        super().__init__(encoders, decoders)

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            '--encoder-embed-dim', type=int, metavar='N',
            help='dimensionality of the encoder embeddings',
        )
        parser.add_argument(
            '--encoder-hidden-dim', type=int, metavar='N',
            help='dimensionality of the encoder hidden state',
        )
        parser.add_argument(
            '--encoder-dropout', type=float, default=0.1,
            help='encoder dropout probability',
        )
        parser.add_argument(
            '--encoder-num-layers', type=int, metavar='N',
            help='number of encoder layers',
        )
        parser.add_argument(
            '--decoder-embed-dim', type=int, metavar='N',
            help='dimensionality of the decoder embeddings',
        )
        parser.add_argument(
            '--decoder-hidden-dim', type=int, metavar='N',
            help='dimensionality of the decoder hidden state',
        )
        parser.add_argument(
            '--decoder-dropout', type=float, default=0.1,
            help='decoder dropout probability',
        )
        parser.add_argument(
            '--lang-embeddings', action='store_true',
            help='whether to use language embeddings in the decoder',
        )
        parser.add_argument(
            '--lang-embed-dim', type=int, metavar='N',
            help='dimensionality of the language id embeddings',
        )

    @classmethod
    def build_model(cls, args, task):
        assert isinstance(task, TranslationLaserTask)

        src_langs = [lang_pair.split('-')[0] for lang_pair in task.model_lang_pairs]
        tgt_langs = [lang_pair.split('-')[1] for lang_pair in task.model_lang_pairs]

        # shared encoder/decoder
        encoder = LaserLSTMEncoder(
            task.dicts[src_langs[0]],
            embed_dim=args.encoder_embed_dim,
            hidden_size=args.encoder_hidden_dim,
            num_layers=args.encoder_num_layers,
            bidirectional=True,
            dropout_in=args.encoder_dropout,
            dropout_out=args.encoder_dropout,
        )
        decoder = LaserLSTMDecoder(
            task.dicts[tgt_langs[0]],
            encoder_output_units=args.encoder_hidden_dim * 2,
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_hidden_dim,
            dropout_in=args.decoder_dropout,
            dropout_out=args.decoder_dropout,
            num_langs=len(task.langs) if args.lang_embeddings else 0,
            lang_embed_dim=args.lang_embed_dim,
        )

        encoders, decoders = OrderedDict(), OrderedDict()
        for lang_pair, src, tgt in zip(task.model_lang_pairs, src_langs, tgt_langs):
            encoders[lang_pair] = encoder
            decoders[lang_pair] = decoder
        return LaserLSTMModel(encoders, decoders)

    def load_state_dict(self, state_dict, strict=True, args=None):
        state_dict_subset = state_dict.copy()
        for k, _ in state_dict.items():
            assert k.startswith('models.')
            lang_pair = k.split('.')[1]
            if lang_pair not in self.models:
                del state_dict_subset[k]
        super().load_state_dict(state_dict_subset, strict=strict, args=args)


@register_model_architecture('laser_lstm', 'laser_lstm')
def base_architecture(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_hidden_dim = getattr(args, 'encoder_hidden_dim', 256)
    args.encoder_num_layers = getattr(args, 'encoder_num_layers', 1)
    args.encoder_dropout = getattr(args, 'encoder_dropout', 0.1)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_hidden_dim = getattr(args, 'decoder_hidden_dim', 256)
    args.decoder_dropout = getattr(args, 'decoder_dropout', 0.1)
    args.lang_embeddings = getattr(args, 'lang_embeddings', False)
    args.lang_embed_dim = getattr(args, 'lang_embed_dim', 32)


@register_model_architecture('laser_lstm', 'laser_lstm_artetxe')
def laser_lstm_artetxe(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 320)
    args.encoder_hidden_dim = getattr(args, 'encoder_hidden_dim', 512)
    args.encoder_num_layers = getattr(args, 'encoder_num_layers', 5)
    args.encoder_dropout = getattr(args, 'encoder_dropout', 0.1)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 320)
    args.decoder_hidden_dim = getattr(args, 'decoder_hidden_dim', 2048)
    args.decoder_dropout = getattr(args, 'decoder_dropout', 0.1)
    args.lang_embeddings = getattr(args, 'lang_embeddings', True)
    args.lang_embed_dim = getattr(args, 'lang_embed_dim', 32)
    base_architecture(args)
