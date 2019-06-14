import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.models import (
    FairseqEncoderDecoderModel,
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


class LaserLSTMEncoder(LSTMEncoder):

    def forward(self, src_tokens, src_lengths, **kwargs):
        encoder_out = super().forward(src_tokens, src_lengths)
        encoder_padding_mask = encoder_out['encoder_padding_mask']
        x, final_hiddens, final_cells = encoder_out['encoder_out']

        if encoder_padding_mask is not None:

            encoder_padding_mask = src_tokens.eq(self.padding_idx).t()

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
        hidden_size=512, lang_num_embeddings=0, lang_embed_dim=32,
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
        if lang_num_embeddings > 0:
            self.embed_lang = Embedding(
                num_embeddings=lang_num_embeddings,
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

    def forward(self, prev_output_tokens, encoder_out, trg_segments=None, incremental_state=None):
        sentemb = encoder_out['sentemb']

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
        bsz, seqlen = prev_output_tokens.size()

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
        # optional language ID embedding
        if trg_segments is not None:
            x_lang = self.embed_lang(trg_segments)
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
class LaserLSTMModel(FairseqEncoderDecoderModel):

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

    @classmethod
    def build_model(cls, args, task):
        # Initialize our Encoder and Decoder.
        encoder = LaserLSTMEncoder(
            task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            hidden_size=args.encoder_hidden_dim,
            dropout_in=args.encoder_dropout,
            dropout_out=args.encoder_dropout,
        )
        decoder = LaserLSTMDecoder(
            task.target_dictionary,
            encoder_output_units=args.encoder_hidden_dim,
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_hidden_dim,
            dropout_in=args.decoder_dropout,
            dropout_out=args.decoder_dropout,
        )
        model = LaserLSTMModel(encoder, decoder)

        # Print the model architecture.
        print(model)

        return model

    def forward(self, src_tokens, src_lengths, prev_output_tokens, **kwargs):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, **kwargs)
        return decoder_out


@register_model_architecture('laser_lstm', 'laser_lstm')
def laser_lstm(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_hidden_dim = getattr(args, 'encoder_hidden_dim', 256)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_hidden_dim = getattr(args, 'decoder_hidden_dim', 256)
