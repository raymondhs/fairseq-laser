#!/usr/bin/env python3 -u
"""
Embed sentences with a trained model. Batches data on-the-fly.
"""

from collections import namedtuple
import fileinput

import torch
import numpy as np

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.data import encoders


Batch = namedtuple('Batch', 'ids src_tokens src_lengths')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')


def buffered_read(input, buffer_size):
    buffer = []
    with fileinput.input(files=[input], openhook=fileinput.hook_encoded("utf-8")) as h:
        for src_str in h:
            buffer.append(src_str.strip())
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(lines, args, task, max_positions, encode_fn):
    tokens = [
        task.source_dictionary.encode_line(
            encode_fn(src_str), add_if_not_exist=False
        ).long()
        for src_str in lines
    ]
    lengths = torch.LongTensor([t.numel() for t in tokens])
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(tokens, lengths),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            ids=batch['id'],
            src_tokens=batch['net_input']['src_tokens'], src_lengths=batch['net_input']['src_lengths'],
        )


def main(args):
    utils.import_user_module(args)

    if args.buffer_size < 1:
        args.buffer_size = 1
    if args.max_tokens is None and args.max_sentences is None:
        args.max_sentences = 1

    assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
        '--max-sentences/--batch-size cannot be larger than --buffer-size'

    print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(args)

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(':'),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    model = models[0] # Just a single model

    # Handle tokenization and BPE
    tokenizer = encoders.build_tokenizer(args)
    bpe = encoders.build_bpe(args)

    def encode_fn(x):
        if tokenizer is not None:
            x = tokenizer.encode(x)
        if bpe is not None:
            x = bpe.encode(x)
        return x

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        *[model.max_positions() for model in models]
    )

    fout = open(args.output_file, mode='wb')
    if args.buffer_size > 1:
        print('| Sentence buffer size:', args.buffer_size)
    start_id = 0
    for inputs in buffered_read(args.input, args.buffer_size):
        indices = []
        results = []
        for batch in make_batches(inputs, args, task, max_positions, encode_fn):
            src_tokens = batch.src_tokens
            src_lengths = batch.src_lengths
            if use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()
            
            model.eval()
            embeddings = model.encoder(src_tokens, src_lengths)['sentemb']
            embeddings = embeddings.detach().cpu().numpy()
            for i, (id, emb) in enumerate(zip(batch.ids.tolist(), embeddings)):
                indices.append(id)
                results.append(emb)
        np.vstack(results)[np.argsort(indices)].tofile(fout)

        # update running id counter
        start_id += len(inputs)
    fout.close()


def cli_main():
    parser = options.get_generation_parser(interactive=True)
    parser.add_argument('--output-file', required=True,
                        help='Output sentence embeddings')
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
