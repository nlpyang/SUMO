from models.encoder import TransformerInterEncoder, StructuredEncoder
from models.optimizers import  Optimizer
import torch.nn as nn
from torch.nn.init import xavier_uniform_

import torch


def build_optim(args, model, checkpoint):
    """ Build optimizer """
    saved_optimizer_state_dict = None

    if args.train_from != '':
        optim = checkpoint['optim']
        saved_optimizer_state_dict = optim.optimizer.state_dict()
    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method=args.decay_method,
            warmup_steps=args.warmup_steps, model_size=args.hidden_size)

    # Stage 1:
    # Essentially optim.set_parameters (re-)creates and optimizer using
    # model.paramters() as parameters that will be stored in the
    # optim.optimizer.param_groups field of the torch optimizer class.
    # Importantly, this method does not yet load the optimizer state, as
    # essentially it builds a new optimizer with empty optimizer state and
    # parameters from the model.
    optim.set_parameters(list(model.named_parameters()))

    if args.train_from != '':
        # Stage 2: In this stage, which is only performed when loading an
        # optimizer from a checkpoint, we load the saved_optimizer_state_dict
        # into the re-created optimizer, to set the optim.optimizer.state
        # field, which was previously empty. For this, we use the optimizer
        # state saved in the "saved_optimizer_state_dict" variable for
        # this purpose.
        # See also: https://github.com/pytorch/pytorch/issues/2830
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        # Convert back the state values to cuda type if applicable
        if args.visible_gpu != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        # We want to make sure that indeed we have a non-empty optimizer state
        # when we loaded an existing model. This should be at least the case
        # for Adam, which saves "exp_avg" and "exp_avg_sq" state
        # (Exponential moving average of gradient and squared gradient values)
        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    return optim




class Summarizer(nn.Module):
    def __init__(self, args, word_padding_idx, vocab_size, device, checkpoint=None, multigpu=False):
        self.multigpu = multigpu
        super(Summarizer, self).__init__()
        self.vocab_size = vocab_size
        self.device = device

        src_embeddings = torch.nn.Embedding(self.vocab_size, args.emb_size, padding_idx=word_padding_idx)
        if(args.structured):
            self.encoder = StructuredEncoder(args.hidden_size, args.ff_size, args.heads, args.dropout, src_embeddings,
                                                   args.local_layers, args.inter_layers)
        else:
            self.encoder = TransformerInterEncoder(args.hidden_size, args.ff_size, args.heads, args.dropout, src_embeddings,
                                                   args.local_layers, args.inter_layers)
        if checkpoint is not None:
            # checkpoint['model']
            keys = list(checkpoint['model'].keys())
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            for p in self.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)

        self.to(device)

    def forward(self, src, labels, src_lengths):
        sent_scores, mask_block = self.encoder(src)

        return sent_scores, mask_block
