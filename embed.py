#!/usr/bin/env python3 -u
"""
Embed sentences with a trained model. Batches data on-the-fly.
"""

from collections import namedtuple
import fileinput

import torch

from fairseq import checkpoint_utils, options, tasks, utils

import numpy as np

Batch = namedtuple('Batch', 'ids src_tokens src_lengths')


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
    model = models[0] # Just a single model

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    if use_cuda:
        model.cuda()

    if args.spm_model:
        import sentencepiece as spm
        # Load SentencePiece model
        sp = spm.SentencePieceProcessor()
        sp.Load(args.spm_model)
        def encode_spm(l):
            result = ' '.join(sp.EncodeAsPieces(l))
            #print("{} -> {}".format(l, result))
            return result
        encode_fn = encode_spm
    else:
        encode_fn = lambda x: x

    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        *[model.max_positions() for model in models]
    )

    fout = open(args.output_file, mode='wb')
    if args.buffer_size > 1:
        print('| Sentence buffer size:', args.buffer_size)
    print('| Reading input sentence from stdin')
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
    parser.add_argument('--spm-model',
                        help='(optional) Path to SentencePiece model')
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
