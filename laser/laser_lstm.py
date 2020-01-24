from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.lstm import (
    Embedding,
    Linear,
    LSTM,
    LSTMCell,
    LSTMModel,
)

@register_model('laser')
class LaserModel(FairseqEncoderDecoderModel):
    """Laser Encoder-Decoder implementation adapted from:
    https://github.com/pytorch/fairseq/blob/master/fairseq/models/lstm.py
    https://github.com/facebookresearch/LASER/blob/master/source/embed.py
    https://github.com/transducens/LASERtrain/blob/master/fairseq-modules/multilingual_lstm_laser.py
    """
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        LSTMModel.add_args(parser)
        # fmt: off
        parser.add_argument('--lang-embedding-size', type=int, default=32,
                            help='language embedding dimension')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        from .laser_task import TranslationLaserTask
        assert isinstance(task, TranslationLaserTask)

        shared_dict = task.dicts[task.langs[0]]
        if any(task.dicts[lang] != shared_dict for lang in task.langs):
            raise ValueError('This model requires a joined dictionary.')

        # make sure that all args are properly defaulted (in case there are any new ones)
        base_architecture(args)
        
        # Languages index: lang codes into integers
        lang_dictionary = {
            task.langs[i] : i for i in range(len(task.langs))
        }

        encoder = LaserEncoder(
            dictionary=task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            hidden_size=args.encoder_hidden_size,
            num_layers=args.encoder_layers,
            bidirectional=args.encoder_bidirectional,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
        )
        decoder = LaserDecoder(
            dictionary=task.target_dictionary,
            lang_dictionary=lang_dictionary,
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_hidden_size,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            attention=False,
            encoder_output_units=int(args.encoder_hidden_size)*2,
            lang_embedding_size=args.lang_embedding_size,
        )
        return cls(encoder, decoder)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, decoder_lang, **kwargs):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, lang=decoder_lang, **kwargs)
        return decoder_out


class LaserEncoder(FairseqEncoder):

    def __init__(
            self, dictionary, embed_dim=320, hidden_size=512, num_layers=1, bidirectional=False,
            left_pad=True, padding_value=0., dropout_in=0.1, dropout_out=0.1
    ):
        super().__init__(dictionary)

        self.num_layers = num_layers
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        num_embeddings = len(dictionary)
        self.padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.dropout_out if num_layers > 1 else 0.,
            bidirectional=bidirectional,
        )
        self.left_pad = left_pad
        self.padding_value = padding_value

        self.output_units = hidden_size
        if bidirectional:
            self.output_units *= 2

    def forward(self, src_tokens, src_lengths):
        if self.left_pad:
            # convert left-padding to right-padding
            src_tokens = utils.convert_padding_direction(
                src_tokens,
                self.padding_idx,
                left_to_right=True,
            )

        bsz, seqlen = src_tokens.size()

        # embed tokens
        x = self.embed_tokens(src_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # pack embedded source tokens into a PackedSequence
        packed_x = nn.utils.rnn.pack_padded_sequence(x, src_lengths.data.tolist())

        # apply LSTM
        if self.bidirectional:
            state_size = 2 * self.num_layers, bsz, self.hidden_size
        else:
            state_size = self.num_layers, bsz, self.hidden_size
        h0 = x.data.new(*state_size).zero_()
        c0 = x.data.new(*state_size).zero_()
        packed_outs, (final_hiddens, final_cells) = self.lstm(packed_x, (h0, c0))

        # unpack outputs and apply dropout
        x, _ = nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=self.padding_value)
        x = F.dropout(x, p=self.dropout_out, training=self.training)
        assert list(x.size()) == [seqlen, bsz, self.output_units]

        if self.bidirectional:
            def combine_bidir(outs):
                return torch.cat([
                    torch.cat([outs[2 * i], outs[2 * i + 1]], dim=0).view(1, bsz, self.output_units)
                    for i in range(self.num_layers)
                ], dim=0)

            final_hiddens = combine_bidir(final_hiddens)
            final_cells = combine_bidir(final_cells)

        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()

        # Set padded outputs to -inf so they are not selected by max-pooling
        padding_mask = src_tokens.eq(self.padding_idx).t().unsqueeze(-1)
        if padding_mask.any():
            x = x.float().masked_fill_(padding_mask, float('-inf')).type_as(x)

        # Build the sentence embedding by max-pooling over the encoder outputs
        sentemb = x.max(dim=0)[0]

        return {
            'sentemb': sentemb,
            'encoder_out': (x, final_hiddens, final_cells),
            'encoder_padding_mask': encoder_padding_mask if encoder_padding_mask.any() else None
        }


class LaserDecoder(FairseqIncrementalDecoder):

    def __init__(
        self, dictionary, lang_dictionary, embed_dim=512, hidden_size=512, out_embed_dim=512,
        num_layers=1, dropout_in=0.1, dropout_out=0.1, attention=True,
        encoder_output_units=512, pretrained_embed=None,
        share_input_output_embed=False, adaptive_softmax_cutoff=None,
        lang_embedding_size=32
    ):
        super().__init__(dictionary)
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.hidden_size = hidden_size
        self.share_input_output_embed = share_input_output_embed
        self.lang_embedding_size = lang_embedding_size
        self.lang_dictionary = lang_dictionary
        self.embed_langs = nn.Embedding(len(lang_dictionary), lang_embedding_size)
        self.need_attn = False

        self.adaptive_softmax = None
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)

        self.encoder_output_units = encoder_output_units
        self.encoder_hidden_proj = Linear(encoder_output_units, hidden_size)
        self.encoder_cell_proj = Linear(encoder_output_units, hidden_size)
        
        input_size = hidden_size + embed_dim + lang_embedding_size + encoder_output_units
        self.layers = nn.ModuleList([
            LSTMCell(
                input_size=input_size if layer == 0 else hidden_size,
                hidden_size=hidden_size,
            )
            for layer in range(num_layers)
        ])
        
        self.attention = None
        if hidden_size != out_embed_dim:
            self.additional_fc = Linear(hidden_size, out_embed_dim)
        if not self.share_input_output_embed:
            self.fc_out = Linear(out_embed_dim, num_embeddings, dropout=dropout_out)

    def forward(self, prev_output_tokens, encoder_out, lang, incremental_state=None):
        x, attn_scores = self.extract_features(
            prev_output_tokens, encoder_out, lang, incremental_state
        )
        return self.output_layer(x), attn_scores

    def extract_features(
        self, prev_output_tokens, encoder_out, lang, incremental_state=None
    ):
        """
        Similar to *forward* but only return features.
        """
        encoder_sentemb = encoder_out['sentemb']
        encoder_padding_mask = encoder_out['encoder_padding_mask']
        encoder_out = encoder_out['encoder_out']

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
        bsz, seqlen = prev_output_tokens.size()

        # get outputs from encoder
        encoder_outs, encoder_hiddens, encoder_cells = encoder_out[:3]
        srclen = encoder_outs.size(0)

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # embed language
        lang_tensor = torch.LongTensor(
            [self.lang_dictionary[lang]] * bsz
        ).to(device=prev_output_tokens.device)
        l = self.embed_langs(lang_tensor)

        # initialize previous states (or get from cache during incremental generation)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is not None:
            prev_hiddens, prev_cells, input_feed = cached_state
        else:
            num_layers = len(self.layers)
            prev_hiddens = [encoder_sentemb for i in range(num_layers)]
            prev_cells = [encoder_sentemb for i in range(num_layers)]
            prev_hiddens = [self.encoder_hidden_proj(x) for x in prev_hiddens]
            prev_cells = [self.encoder_cell_proj(x) for x in prev_cells]
            input_feed = x.new_zeros(bsz, self.hidden_size)

        outs = []
        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            input = torch.cat((x[j, :, :], encoder_sentemb, input_feed, l), dim=1)

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                input = F.dropout(hidden, p=self.dropout_out, training=self.training)

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            out = hidden
            out = F.dropout(out, p=self.dropout_out, training=self.training)

            # input feeding
            input_feed = out

            # save final output
            outs.append(out)

        # cache previous states (no-op except during incremental generation)
        utils.set_incremental_state(
            self, incremental_state, 'cached_state',
            (prev_hiddens, prev_cells, input_feed),
        )

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hidden_size)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        if hasattr(self, 'additional_fc') and self.adaptive_softmax is None:
            x = self.additional_fc(x)
            x = F.dropout(x, p=self.dropout_out, training=self.training)

        return x, None

    def output_layer(self, x):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = self.fc_out(x)
        return x

    def reorder_incremental_state(self, incremental_state, new_order):
        super().reorder_incremental_state(incremental_state, new_order)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is None:
            return

        def reorder_state(state):
            if isinstance(state, list):
                return [reorder_state(state_i) for state_i in state]
            return state.index_select(0, new_order)

        new_state = tuple(map(reorder_state, cached_state))
        utils.set_incremental_state(self, incremental_state, 'cached_state', new_state)


@register_model_architecture('laser', 'laser')
def base_architecture(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_freeze_embed = getattr(args, 'encoder_freeze_embed', False)
    args.encoder_hidden_size = getattr(args, 'encoder_hidden_size', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', 1)
    args.encoder_bidirectional = getattr(args, 'encoder_bidirectional', True)
    args.encoder_dropout_in = getattr(args, 'encoder_dropout_in', args.dropout)
    args.encoder_dropout_out = getattr(args, 'encoder_dropout_out', args.dropout)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_freeze_embed = getattr(args, 'decoder_freeze_embed', False)
    args.decoder_hidden_size = getattr(args, 'decoder_hidden_size', 2048)
    args.decoder_layers = getattr(args, 'decoder_layers', 1)
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 2048)
    args.decoder_attention = getattr(args, 'decoder_attention', False)
    args.decoder_dropout_in = getattr(args, 'decoder_dropout_in', args.dropout)
    args.decoder_dropout_out = getattr(args, 'decoder_dropout_out', args.dropout)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
