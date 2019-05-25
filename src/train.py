#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse
import glob
import os
import time

import sentencepiece

from models import data_loader, model_builder
from models.data_loader import load_dataset
from models.model_builder import Summarizer
from models.trainer import build_trainer
from others.logging import logger, init_logger
import torch
import random

model_flags = ['hidden_size', 'ff_size', 'heads', 'emb_size', 'local_layers', 'inter_layers','structured']

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')




def wait_and_validate(args, device_id):
    timestep = 0
    if (args.test_all):
        cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
        cp_files.sort(key=os.path.getmtime)
        xent_lst = []
        for i, cp in enumerate(cp_files):
            step = int(cp.split('.')[-2].split('_')[-1])
            xent = validate(args, device_id, cp, step)
            xent_lst.append((xent, cp))
            max_step = xent_lst.index(min(xent_lst))
            if (i - max_step > 10):
                break
        xent_lst = sorted(xent_lst, key=lambda x: x[0])[:5]
        logger.info('PPL %s' % str(xent_lst))
        for xent, cp in xent_lst:
            step = int(cp.split('.')[-2].split('_')[-1])
            test(args, device_id, cp, step)
    else:
        while (True):
            cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            if (cp_files):
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)
                if (not os.path.getsize(cp) > 0):
                    time.sleep(60)
                    continue
                if (time_of_cp > timestep):
                    timestep = time_of_cp
                    step = int(cp.split('.')[-2].split('_')[-1])
                    validate(args, device_id, cp, step)
                    test(args, device_id, cp, step)

            cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            if (cp_files):
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)
                if (time_of_cp > timestep):
                    continue
            else:
                time.sleep(300)


def validate(args, device_id, pt, step):
    device = "cpu" if args.visible_gpu == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    spm = sentencepiece.SentencePieceProcessor()
    spm.Load(args.vocab_path)
    word_padding_idx = spm.PieceToId('<PAD>')
    vocab_size = len(spm)
    model = Summarizer(args, word_padding_idx, vocab_size, device, checkpoint)
    model.eval()

    valid_iter =data_loader.Dataloader(args, load_dataset(args, 'valid', shuffle=False), {'PAD': word_padding_idx},
                                  args.batch_size, device,
                                  shuffle=False, is_test=False)
    trainer = build_trainer(args, device_id, model, None)
    stats = trainer.validate(valid_iter)
    trainer._report_step(0, step, valid_stats=stats)
    return stats.xent()

def test(args, device_id, pt, step):
    device = "cpu" if args.visible_gpu == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    spm = sentencepiece.SentencePieceProcessor()
    spm.Load(args.vocab_path)
    word_padding_idx = spm.PieceToId('<PAD>')
    vocab_size = len(spm)
    model = Summarizer(args, word_padding_idx, vocab_size, device, checkpoint)
    model.eval()

    test_iter =data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False), {'PAD': word_padding_idx},
                                  args.batch_size, device,
                                  shuffle=False, is_test=True)
    trainer = build_trainer(args, device_id, model, None)
    trainer.test(test_iter,step)




def train(args, device_id):
    init_logger(args.log_file)

    if args.train_from != '':
        logger.info('Loading checkpoint from %s' % args.train_from)
        checkpoint = torch.load(args.train_from,
                                map_location=lambda storage, loc: storage)
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if (k in model_flags):
                setattr(args, k, opt[k])
    else:
        checkpoint = None

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True


    spm = sentencepiece.SentencePieceProcessor()
    spm.Load(args.vocab_path)
    word_padding_idx = spm.PieceToId('<PAD>')
    vocab_size = len(spm)


    def train_iter_fct():
        # return data_loader.AbstractiveDataloader(load_dataset('train', True), symbols, FLAGS.batch_size, device, True)
        return data_loader.Dataloader(args, load_dataset(args, 'train', shuffle=True), {'PAD':word_padding_idx}, args.batch_size, device,
                                                 shuffle=True, is_test=False)

    model = Summarizer(args, word_padding_idx, vocab_size, device, checkpoint)
    optim = model_builder.build_optim(args, model, checkpoint)
    logger.info(model)
    trainer = build_trainer(args, device_id, model, optim)
    #
    trainer.train(train_iter_fct, args.train_steps)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()



    parser.add_argument("-mode", default='train', type=str)
    parser.add_argument("-onmt_path", default='../data/onmt_data/cnndm')
    parser.add_argument("-data_path", default='../data/')
    parser.add_argument("-raw_path", default='../line_data')
    parser.add_argument("-vocab_path", default='../data/spm.cnndm.model')
    parser.add_argument("-model_path", default='../models/')
    parser.add_argument("-result_path", default='../results/cnndm')
    parser.add_argument("-temp_dir", default='../temp')

    parser.add_argument("-batch_size", default=10000, type=int)
    parser.add_argument('-min_nsents', default=3, type=int)
    parser.add_argument('-max_nsents', default=100, type=int)
    parser.add_argument('-min_src_ntokens', default=5, type=int)
    parser.add_argument('-max_src_ntokens', default=200, type=int)


    parser.add_argument("-structured", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-hidden_size", default=128, type=int)
    parser.add_argument("-ff_size", default=512, type=int)
    parser.add_argument("-heads", default=8, type=int)
    parser.add_argument("-emb_size", default=128, type=int)
    parser.add_argument("-local_layers", default=5, type=int)
    parser.add_argument("-inter_layers", default=2, type=int)

    parser.add_argument("-dropout", default=0.2, type=float)
    parser.add_argument("-optim", default='adam', type=str)
    parser.add_argument("-lr", default=0.15, type=float)
    parser.add_argument("-beta1", default= 0.9, type=float)
    parser.add_argument("-beta2", default=0.999, type=float)
    parser.add_argument("-decay_method", default='', type=str)
    parser.add_argument("-warmup_steps", default=8000, type=int)
    parser.add_argument("-max_grad_norm", default=0, type=float)

    parser.add_argument("-save_checkpoint_steps", default=5, type=int)
    parser.add_argument("-accum_count", default=1, type=int)
    parser.add_argument("-report_every", default=10, type=int)
    parser.add_argument("-train_steps", default=1000, type=int)


    parser.add_argument('-visible_gpu', default='-1', type=str)
    parser.add_argument('-gpu_ranks', default=[0], type=list)
    parser.add_argument('-log_file', default='../logs/cnndm.log')
    parser.add_argument('-dataset', default='')
    parser.add_argument('-seed', default=666, type=int)

    parser.add_argument("-test_all", type=str2bool, nargs='?',const=True,default=False)
    parser.add_argument("-train_from", default='')
    parser.add_argument("-test_from", default='')
    parser.add_argument("-report_rouge", type=str2bool, nargs='?',const=True,default=True)

    args = parser.parse_args()
    print(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpu

    init_logger(args.log_file)
    device = "cpu" if args.visible_gpu == '-1' else "cuda"
    device_id = 0 if device == "cuda" else -1

    if (args.mode == 'train'):
        train(args, device_id)
    elif (args.mode == 'validate'):
        wait_and_validate(args, device_id)
    elif (args.mode == 'test'):
        test(args, device_id, args.test_from, 0)
