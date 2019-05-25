import codecs
import shutil
import time
import torch
import os
from others import pyrouge
from models.reporter import ReportMgr
from models.stats import Statistics
from others.logging import logger
from tensorboardX import SummaryWriter
import numpy as np


def _get_ngrams(n, text):
    ngram_set = set()
    text_length = len(text)
    max_index_ngram_start = text_length - n
    for i in range(max_index_ngram_start + 1):
        ngram_set.add(tuple(text[i:i + n]))
    return ngram_set


def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' or 'generator' in name:
            dec += param.nelement()
    return n_params, enc, dec


def build_trainer(args, device_id, model,
                  optim):
    grad_accum_count = args.accum_count
    n_gpu = 1
    if device_id < 0:
        n_gpu = 0

    gpu_rank = 0

    tensorboard_log_dir = args.model_path

    writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")

    report_manager = ReportMgr(args.report_every, start_time=-1, tensorboard_writer=writer)

    trainer = Trainer(args, model, optim, grad_accum_count, n_gpu, gpu_rank, report_manager)

    n_params, enc, dec = _tally_parameters(model)
    logger.info('encoder: %d' % enc)
    logger.info('decoder: %d' % dec)
    logger.info('* number of parameters: %d' % n_params)

    return trainer


def process(temp_dir,candidates, references):
    cnt = len(candidates)
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    tmp_dir = os.path.join(temp_dir, "rouge-tmp-{}".format(current_time))
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
        os.mkdir(tmp_dir + "/candidate")
        os.mkdir(tmp_dir + "/reference")
    try:

        for i in range(cnt):
            if len(references[i]) < 1:
                continue
            with open(tmp_dir + "/candidate/cand.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(candidates[i])
            with open(tmp_dir + "/reference/ref.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(references[i])
        r = pyrouge.Rouge155(temp_dir = temp_dir)
        r.model_dir = tmp_dir + "/reference/"
        r.system_dir = tmp_dir + "/candidate/"
        r.model_filename_pattern = 'ref.#ID#.txt'
        r.system_filename_pattern = r'cand.(\d+).txt'
        rouge_results = r.convert_and_evaluate()
        print(rouge_results)
        results_dict = r.output_to_dict(rouge_results)
    finally:
        pass
        if os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir)
    return results_dict

def test_rouge(temp_dir, cand, ref, num_processes):
    candidates = [line.strip() for line in cand]
    references = [line.strip() for line in ref]

    print(len(candidates))
    print(len(references))
    assert len(candidates) == len(references)
    results = process(temp_dir,candidates,references)
    return results

def rouge_results_to_str(results_dict):
    return ">> ROUGE-F(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\nROUGE-R(1/2/3/l): {:.2f}/{:.2f}/{:.2f}\n".format(
        results_dict["rouge_1_f_score"] * 100,
        results_dict["rouge_2_f_score"] * 100,
        results_dict["rouge_l_f_score"] * 100,
        results_dict["rouge_1_recall"] * 100,
        results_dict["rouge_2_recall"] * 100,
        results_dict["rouge_l_recall"] * 100

        # ,results_dict["rouge_su*_f_score"] * 100
    )
class Trainer(object):

    def __init__(self,  args, model,  optim,
                  grad_accum_count=1, n_gpu=1, gpu_rank=1,
                  report_manager=None):
        # Basic attributes.
        self.args = args
        self.save_checkpoint_steps = args.save_checkpoint_steps
        self.model = model
        self.optim = optim
        self.grad_accum_count = grad_accum_count
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.report_manager = report_manager


        self.loss = torch.nn.BCELoss(reduction='none')
        assert grad_accum_count > 0
        # Set model in training mode.
        self.model.train()

    def train(self, train_iter_fct, train_steps, valid_iter_fct=None, valid_steps=-1):
        logger.info('Start training...')

        step = self.optim._step + 1
        true_batchs = []
        accum = 0
        normalization = 0
        train_iter = train_iter_fct()

        total_stats = Statistics()
        report_stats = Statistics()
        self._start_report_manager(start_time=total_stats.start_time)

        while step <= train_steps:

            reduce_counter = 0
            for i, batch in enumerate(train_iter):
                if self.n_gpu == 0 or (i % self.n_gpu == self.gpu_rank):

                    true_batchs.append(batch)
                    normalization += batch.batch_size
                    accum += 1
                    if accum == self.grad_accum_count:
                        reduce_counter += 1
                        self._gradient_accumulation(
                            true_batchs, normalization, total_stats,
                            report_stats)

                        report_stats = self._maybe_report_training(
                            step, train_steps,
                            self.optim.learning_rate,
                            report_stats)

                        true_batchs = []
                        accum = 0
                        normalization = 0
                        if (step % self.save_checkpoint_steps == 0 and self.gpu_rank == 0):
                            self._save(step)

                        step += 1
                        if step > train_steps:
                            break
            train_iter = train_iter_fct()

        return total_stats

    def validate(self, valid_iter):
        # Set model in validating mode.
        self.model.eval()
        stats = Statistics()

        with torch.no_grad():
            for batch in valid_iter:
                src = batch.src
                src_lengths = batch.src_length
                labels = batch.labels

                if(self.args.structured):
                    roots, mask = self.model(src, labels, src_lengths)
                    r = torch.clamp(roots[-1], 1e-5, 1 - 1e-5)
                    loss = self.loss(r, labels)

                else:
                    sent_scores, mask = self.model(src, labels, src_lengths)
                    loss = self.loss(sent_scores, labels)
                loss = (loss * mask.float()).sum()
                batch_stats = Statistics(float(loss.cpu().data.numpy()), len(labels))

                stats.update(batch_stats)
            return stats


    def test(self, test_iter, step):
        def _block_tri(c, p):
            tri_c = _get_ngrams(3, c.split())
            for s in p:
                tri_s = _get_ngrams(3, s.split())
                if len(tri_c.intersection(tri_s))>0:
                    return True
            return False

        self.model.eval()
        gold_path = '%s_step%d.candidate'%(self.args.result_path,step)
        can_path = '%s_step%d.gold' % (self.args.result_path, step)
        with open(gold_path, 'w') as save_pred:
            with open(can_path, 'w') as save_gold:
                with torch.no_grad():
                    for batch in test_iter:
                        src = batch.src
                        src_lengths = batch.src_length
                        labels = batch.labels
                        gold = []
                        pred = []
                        if(self.args.structured):
                            roots, mask = self.model(src, labels, src_lengths)
                            sent_scores = roots[-1] + mask.float()
                        else:
                            sent_scores, mask = self.model(src, labels, src_lengths)
                            sent_scores = sent_scores+mask.float()
                        sent_scores  = sent_scores.cpu().data.numpy()
                        selected_ids = np.argsort(-sent_scores,1)
                        # selected_ids = np.sort(selected_ids,1)
                        for i, idx in enumerate(selected_ids):
                            _pred = []
                            if(len(batch.src_str[i])==0):
                                continue
                            for j in selected_ids[i][:len(batch.src_str[i])]:
                                candidate = batch.src_str[i][j].strip()
                                if(not _block_tri(candidate,_pred)):
                                    _pred.append(candidate)

                                if(len(_pred)==3):
                                    break
                            pred.append('<q>'.join(_pred))
                            gold.append(batch.tgt_str[i])

                        for i in range(len(gold)):
                            save_gold.write(gold[i].strip()+'\n')
                        for i in range(len(pred)):
                            save_pred.write(pred[i].strip()+'\n')
        if(step!=-1 and self.args.report_rouge):
            rouges = self._report_rouge(gold_path, can_path)
            logger.info('Rouges at step %d \n%s'%(step,rouge_results_to_str(rouges)))


    def _report_rouge(self, gold_path, can_path):
        logger.info("Calculating Rouge")

        candidates = codecs.open(can_path, encoding="utf-8")
        references = codecs.open(gold_path, encoding="utf-8")
        results_dict = test_rouge(self.args.temp_dir, candidates, references, 1)
        return results_dict

    def _gradient_accumulation(self, true_batchs, normalization, total_stats,
                               report_stats):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:
            src = batch.src

            src_lengths = batch.src_length
            labels = batch.labels

            if self.grad_accum_count == 1:
                self.model.zero_grad()


            if(self.args.structured):
                roots, mask = self.model(src, labels, src_lengths)
                loss = 0
                for r in roots:
                    r = torch.clamp(r, 1e-5, 1 - 1e-5)
                    _loss = self.loss(r, labels)
                    _loss = (_loss * mask.float()).sum()
                    loss += _loss
                loss = loss/len(roots)
                (loss / loss.numel()).backward()


            else:
                sent_scores, mask = self.model(src, labels, src_lengths)
                loss = self.loss(sent_scores, labels)
                loss = (loss*mask.float()).sum()
                (loss/loss.numel()).backward()
            # loss.div(float(normalization)).backward()

            batch_stats = Statistics(float(loss.cpu().data.numpy()), normalization)


            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

            # 4. Update the parameters and statistics.
            if self.grad_accum_count == 1:
                self.optim.step()

        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.grad_accum_count > 1:
            self.optim.step()

    def _save(self, step):
        real_model = self.model
        # real_generator = (self.generator.module
        #                   if isinstance(self.generator, torch.nn.DataParallel)
        #                   else self.generator)

        model_state_dict = real_model.state_dict()
        # generator_state_dict = real_generator.state_dict()
        checkpoint = {
            'model': model_state_dict,
            # 'generator': generator_state_dict,
            'opt': self.args,
            'optim': self.optim,
        }
        checkpoint_path = os.path.join(self.args.model_path, 'model_step_%d.pt' % step)
        logger.info("Saving checkpoint %s" % checkpoint_path)
        # checkpoint_path = '%s_step_%d.pt' % (FLAGS.model_path, step)
        if (not os.path.exists(checkpoint_path)):
            torch.save(checkpoint, checkpoint_path)
            return checkpoint, checkpoint_path

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats,
                multigpu=self.n_gpu > 1)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)

    def _maybe_save(self, step):
        """
        Save the model if a model saver is set
        """
        if self.model_saver is not None:
            self.model_saver.maybe_save(step)
