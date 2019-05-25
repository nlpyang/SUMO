import gc
import glob
import random
import torch

from others.logging import logger


class Batch(object):
    def _pad(self, data, height, width, pad_id):
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        rtn_length = [len(d) for d in data]
        rtn_data = rtn_data + [[pad_id] * width] * (height - len(data))
        rtn_length = rtn_length + [0] * (height - len(data))
        return rtn_data, rtn_length

    def _pad2(self, data, width, pad_id):
        rtn_data = [d + [pad_id] * (width - len(d)) for d in data]
        return rtn_data

    def __init__(self, data=None, pad_id=None, device=None, is_test=False):
        """Create a Batch from a list of examples."""
        if data is not None:
            self.batch_size = len(data)
            src = [x[0] for x in data]
            labels = [x[1] for x in data]
            max_nsent = max([len(e) for e in src])
            max_ntoken = max([max([len(p) for p in e]) for e in src])
            labels = self._pad2(labels, max_nsent, 0)
            labels = torch.tensor(labels).float()

            _src = [self._pad(e, max_nsent, max_ntoken, pad_id) for e in src]
            src = torch.stack([torch.tensor(e[0]) for e in _src])  # batch_size, n_block, block_size
            src_length = torch.tensor([sum(e[1]) for e in _src])

            setattr(self, 'src', src.to(device))
            setattr(self, 'src_length', src_length.to(device))
            setattr(self, 'labels', labels.to(device))


            # _tgt = self._pad(tgt, width=max([len(d) for d in tgt]), height=len(tgt), pad_id=pad_id)
            # tgt = torch.tensor(_tgt[0]).transpose(0, 1)  # tgt_len * batch_size
            # setattr(self, 'tgt', tgt.to(device))

            if (is_test):
                src_str = [x[2] for x in data]
                setattr(self, 'src_str', src_str)
                tgt_str = [x[3] for x in data]
                setattr(self, 'tgt_str', tgt_str)

    def __len__(self):
        return self.batch_size


def batch(data, batch_size):
    minibatch, size_so_far = [], 0
    for ex in data:
        minibatch.append(ex)
        size_so_far = simple_batch_size_fn(ex, len(minibatch))
        if size_so_far == batch_size:
            yield minibatch
            minibatch, size_so_far = [], 0
        elif size_so_far > batch_size:
            yield minibatch[:-1]
            minibatch, size_so_far = minibatch[-1:], simple_batch_size_fn(ex, 1)
    if minibatch:
        yield minibatch


def load_dataset(args, corpus_type, shuffle):
    assert corpus_type in ["train", "valid", "test"]

    def _lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, pt_file, len(dataset)))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(args.onmt_path + '.' + corpus_type + '.[0-9]*.pt'))
    if pts:
        if (shuffle):
            random.shuffle(pts)

        for pt in pts:
            yield _lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one inputters.*Dataset, simple!
        pt = args.onmt_path + '.' + corpus_type + '.pt'
        yield _lazy_dataset_loader(pt, corpus_type)


def simple_batch_size_fn(new, count):
    src, labels = new[0], new[1]
    global max_n_sents, max_n_tokens, max_size
    if count == 1:
        max_size = 0
        max_n_sents=0
        max_n_tokens=0
    max_n_sents = max(max_n_sents, len(src))
    max_n_tokens = max(max_n_tokens, max([len(s) for s in src]))
    max_size = max(max_size, max_n_sents*max_n_tokens)
    src_elements = count * max_size
    return src_elements


class Dataloader(object):
    def __init__(self, args, datasets, symbols, batch_size,
                 device, shuffle, is_test):
        self.args = args
        self.datasets = datasets
        self.symbols = symbols
        self.batch_size = batch_size
        self.device = device
        self.shuffle = shuffle
        self.is_test = is_test
        self.cur_iter = self._next_dataset_iterator(datasets)
        assert self.cur_iter is not None

    def __iter__(self):
        dataset_iter = (d for d in self.datasets)
        while self.cur_iter is not None:
            for batch in self.cur_iter:
                yield batch
            self.cur_iter = self._next_dataset_iterator(dataset_iter)


    def _next_dataset_iterator(self, dataset_iter):
        try:
            # Drop the current dataset for decreasing memory
            if hasattr(self, "cur_dataset"):
                self.cur_dataset = None
                gc.collect()
                del self.cur_dataset
                gc.collect()

            self.cur_dataset = next(dataset_iter)
        except StopIteration:
            return None

        return DataIterator(args = self.args,
            dataset=self.cur_dataset, symbols=self.symbols, batch_size=self.batch_size,
            device=self.device, shuffle=self.shuffle, is_test=self.is_test)


class DataIterator(object):
    def __init__(self, args, dataset, symbols, batch_size, device=None, is_test=False,
                 shuffle=True):
        self.args = args
        self.batch_size, self.is_test, self.dataset = batch_size, is_test, dataset
        self.iterations = 0
        self.device = device
        self.shuffle = shuffle

        self.sort_key = lambda x: len(x[1])

        self._iterations_this_epoch = 0

        self.symbols = symbols

    def data(self):
        if self.shuffle:
            random.shuffle(self.dataset)
        xs = self.dataset
        return xs

    def preprocess(self, ex):
        src = ex['src']
        labels = [0]*len(src)
        for l in ex['labels'][0]:
            labels[l] = 1
        idxs =  [i for i,s in enumerate(ex['src']) if (len(s)>self.args.min_src_ntokens)]


        src = [src[i][:self.args.max_src_ntokens] for i in idxs]
        labels = [labels[i] for i in idxs]
        src = src[:self.args.max_nsents]
        labels = labels[:self.args.max_nsents]

        if(len(src)<self.args.min_nsents):
            return None
        if(len(labels)==0):
            return None

        if(self.is_test):
            src_txt = [ex['src_txt'][i] for i in idxs]

            return src, labels, src_txt, ex['tgt_txt']

        return src, labels


    def batch_buffer(self, data, batch_size):
        minibatch, size_so_far = [], 0
        for ex in data:
            if(len(ex['src'])==0):
                continue
            ex = self.preprocess(ex)
            if(ex is None):
                continue
            minibatch.append(ex)
            size_so_far = simple_batch_size_fn(ex, len(minibatch))
            if size_so_far == batch_size:
                yield minibatch
                minibatch, size_so_far = [], 0
            elif size_so_far > batch_size:
                yield minibatch[:-1]
                minibatch, size_so_far = minibatch[-1:], simple_batch_size_fn(ex, 1)
        if minibatch:
            yield minibatch

    def create_batches(self):
        """ Create batches """
        data = self.data()
        for buffer in self.batch_buffer(data, self.batch_size * 100):

            p_batch = sorted(buffer, key=lambda x: max([len(s) for s in x[0]]))
            p_batch = sorted(p_batch, key=lambda x: len(x[0]))
            p_batch = batch(p_batch, self.batch_size)

            p_batch = list(p_batch)
            if (self.shuffle):
                random.shuffle(p_batch)
            for b in p_batch:
                yield b

    def __iter__(self):
        while True:
            self.batches = self.create_batches()
            for idx, minibatch in enumerate(self.batches):
                # fast-forward if loaded from state
                if self._iterations_this_epoch > idx:
                    continue
                self.iterations += 1
                self._iterations_this_epoch += 1
                batch = Batch(minibatch, self.symbols['PAD'], self.device, self.is_test)

                yield batch
            return
