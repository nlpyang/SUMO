""" Multi-Head Attention module """
import math
import torch
import torch.nn as nn


# from onmt.utils.misc import aeq


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_count, model_dim, dropout=0.1, use_final_linear=True):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim,
                                      head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.use_final_linear = use_final_linear
        if (self.use_final_linear):
            self.final_linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None,
                layer_cache=None, type=None, predefined_graph_1=None):
        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                .view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query.
        if layer_cache is not None:
            if type == "self":
                query, key, value = self.linear_query(query), \
                                    self.linear_keys(query), \
                                    self.linear_values(query)

                key = shape(key)
                value = shape(value)

                if layer_cache is not None:
                    device = key.device
                    if layer_cache["self_keys"] is not None:
                        key = torch.cat(
                            (layer_cache["self_keys"].to(device), key),
                            dim=2)
                    if layer_cache["self_values"] is not None:
                        value = torch.cat(
                            (layer_cache["self_values"].to(device), value),
                            dim=2)
                    layer_cache["self_keys"] = key
                    layer_cache["self_values"] = value
            elif type == "context":
                query = self.linear_query(query)
                if layer_cache is not None:
                    if layer_cache["memory_keys"] is None:
                        key, value = self.linear_keys(key), \
                                     self.linear_values(value)
                        key = shape(key)
                        value = shape(value)
                    else:
                        key, value = layer_cache["memory_keys"], \
                                     layer_cache["memory_values"]
                    layer_cache["memory_keys"] = key
                    layer_cache["memory_values"] = value
                else:
                    key, value = self.linear_keys(key), \
                                 self.linear_values(value)
                    key = shape(key)
                    value = shape(value)
        else:
            key = self.linear_keys(key)
            value = self.linear_values(value)
            query = self.linear_query(query)
            key = shape(key)
            value = shape(value)

        query = shape(query)


        # 2) Calculate and scale scores.
        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.

        attn = self.softmax(scores)

        if (not predefined_graph_1 is None):
            attn_masked = attn[:, -1] * predefined_graph_1
            attn_masked = attn_masked / (torch.sum(attn_masked, 2).unsqueeze(2) + 1e-9)

            attn = torch.cat([attn[:, :-1], attn_masked.unsqueeze(1)], 1)

        drop_attn = self.dropout(attn)
        if (self.use_final_linear):
            context = unshape(torch.matmul(drop_attn, value))
            output = self.final_linear(context)
            return output
        else:
            context = torch.matmul(drop_attn, value)
            return context



def _getMatrixTree_multi(scores, root):
    A = scores.exp()
    R = root.exp()

    L = torch.sum(A, 1)
    L = torch.diag_embed(L)
    L = L - A
    LL = L + torch.diag_embed(R)
    LL_inv = torch.inverse(LL)  # batch_l, doc_l, doc_l
    LL_inv_diag = torch.diagonal(LL_inv, 0, 1, 2)
    d0 = R * LL_inv_diag
    LL_inv_diag = torch.unsqueeze(LL_inv_diag, 2)

    _A = torch.transpose(A, 1, 2)
    _A = _A * LL_inv_diag
    tmp1 = torch.transpose(_A, 1, 2)
    tmp2 = A * torch.transpose(LL_inv, 1, 2)

    d = tmp1 - tmp2
    return d, d0


class StructuredAttention(nn.Module):
    def __init__(self, model_dim, dropout=0.1):
        self.model_dim = model_dim

        super(StructuredAttention, self).__init__()

        self.linear_keys = nn.Linear(model_dim, self.model_dim)
        self.linear_query = nn.Linear(model_dim, self.model_dim)
        self.linear_root = nn.Linear(model_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):


        key = self.linear_keys(x)
        query = self.linear_query(x)
        root = self.linear_root(x).squeeze(-1)

        query = query / math.sqrt(self.model_dim)
        scores = torch.matmul(query, key.transpose(1, 2))

        mask = mask.float()
        root = root - mask.squeeze(1) * 50
        root = torch.clamp(root, min=-40)
        scores = scores - mask * 50
        scores = scores - torch.transpose(mask, 1, 2) * 50
        scores = torch.clamp(scores, min=-40)
        # _logits = _logits + (tf.transpose(bias, [0, 2, 1]) - 1) * 40
        # _logits = tf.clip_by_value(_logits, -40, 1e10)

        d, d0 = _getMatrixTree_multi(scores, root)
        attn = torch.transpose(d, 1,2)
        if mask is not None:
            mask = mask.expand_as(scores).byte()
            attn = attn.masked_fill(mask, 0)

        return attn, d0


class MultiHeadedPooling(nn.Module):
    def __init__(self, model_dim):
        self.model_dim = model_dim
        super(MultiHeadedPooling, self).__init__()
        self.linear_keys = nn.Linear(model_dim, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):

        scores = self.linear_keys(x).squeeze(-1)


        if mask is not None:
            scores = scores.masked_fill(1 - mask, -1e18)

        attn = self.softmax(scores).unsqueeze(-1)
        output = torch.sum(attn * x, -2)
        return output
