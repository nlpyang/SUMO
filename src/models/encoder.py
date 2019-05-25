"""
Implementation of "Attention is All You Need"
"""
import math

import torch.nn as nn
import torch
from models.attentions import MultiHeadedAttention, MultiHeadedPooling, StructuredAttention



class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x

class PositionalEncoding(nn.Module):

    def __init__(self, dropout, dim, max_len=5000):
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                              -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        emb = emb * math.sqrt(self.dim)
        if (step):
            emb = emb + self.pe[:, step][:, None, :]

        else:
            emb = emb + self.pe[:, :emb.size(1)]
        emb = self.dropout(emb)
        return emb

    def get_emb(self, emb):
        return self.pe[:, :emb.size(1)]

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(
            heads, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, inputs, mask):
        input_norm = self.layer_norm(inputs)
        mask = mask.unsqueeze(1)
        context = self.self_attn(input_norm, input_norm, input_norm,
                                 mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)



class TransformerInterEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, embeddings,  num_local_layers=0, num_inter_layers=0):
        super(TransformerInterEncoder, self).__init__()
        self.d_model = d_model
        self.num_local_layers = num_local_layers
        self.num_inter_layers = num_inter_layers
        self.embeddings = embeddings
        self.pos_emb = PositionalEncoding(dropout, int(self.embeddings.embedding_dim))
        self.transformer_local = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_local_layers)])
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_inter_layers)])
        self.pooling = MultiHeadedPooling(d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.wo = nn.Linear(d_model,1,bias=True)
        self.sigmoid = nn.Sigmoid()


    def forward(self, src):
        batch_size, n_blocks, n_tokens = src.size()
        emb = self.embeddings(src)
        padding_idx = self.embeddings.padding_idx
        mask_local = 1 - src.data.eq(padding_idx).view(batch_size * n_blocks, n_tokens)
        mask_block = torch.sum(mask_local.view(batch_size, n_blocks, n_tokens), -1) > 0


        local_pos_emb = self.pos_emb.pe[:, :n_tokens].unsqueeze(1).expand(batch_size, n_blocks, n_tokens, self.embeddings.embedding_dim)
        emb = emb * math.sqrt(self.embeddings.embedding_dim)
        emb = emb + local_pos_emb
        emb = self.pos_emb.dropout(emb)


        word_vec = emb.view(batch_size * n_blocks, n_tokens, -1)

        for i in range(self.num_local_layers):
            word_vec = self.transformer_local[i](word_vec, word_vec, 1 - mask_local)  # all_sents * max_tokens * dim


        mask_hier = mask_local[:, :, None].float()
        word_vec = word_vec * mask_hier
        word_vec = self.layer_norm1(word_vec)

        sent_vec = self.pooling(word_vec, mask_local)
        sent_vec = sent_vec.view(batch_size, n_blocks, -1)
        global_pos_emb = self.pos_emb.pe[:, :n_blocks]
        sent_vec = sent_vec+global_pos_emb

        for i in range(self.num_inter_layers):
            sent_vec = self.transformer_inter[i](sent_vec, sent_vec, 1 - mask_block)  # all_sents * max_tokens * dim


        sent_vec = self.layer_norm2(sent_vec)
        sent_scores = self.sigmoid(self.wo(sent_vec))
        sent_scores = sent_scores.squeeze(-1) * mask_block.float()

        return sent_scores, mask_block

class StructuredEncoder(nn.Module):
    def __init__(self, d_model, d_ff, heads, dropout, embeddings,  num_local_layers=0, num_inter_layers=0):
        super(StructuredEncoder, self).__init__()
        self.d_model = d_model
        self.num_local_layers = num_local_layers
        self.num_inter_layers = num_inter_layers
        self.embeddings = embeddings
        self.pos_emb = PositionalEncoding(dropout, int(self.embeddings.embedding_dim))
        self.transformer_local = nn.ModuleList(
            [TransformerEncoderLayer(d_model, heads, d_ff, dropout)
             for _ in range(num_local_layers)])
        self.transformer_inter = nn.ModuleList([TMTLayer(d_model, d_ff, dropout, i) for i in range(num_inter_layers)])
        self.pooling = MultiHeadedPooling(d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1= nn.LayerNorm(d_model, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(d_model, eps=1e-6)
        self.wo = nn.Linear(d_model,1,bias=True)
        self.sigmoid = nn.Sigmoid()


    def forward(self, src):
        """ See :obj:`EncoderBase.forward()`"""
        batch_size, n_blocks, n_tokens = src.size()
        emb = self.embeddings(src)
        padding_idx = self.embeddings.padding_idx
        mask_local = 1 - src.data.eq(padding_idx).view(batch_size * n_blocks, n_tokens)
        mask_block = torch.sum(mask_local.view(batch_size, n_blocks, n_tokens), -1) > 0


        local_pos_emb = self.pos_emb.pe[:, :n_tokens].unsqueeze(1).expand(batch_size, n_blocks, n_tokens, self.embeddings.embedding_dim)
        emb = emb * math.sqrt(self.embeddings.embedding_dim)
        emb = emb + local_pos_emb
        emb = self.pos_emb.dropout(emb)


        word_vec = emb.view(batch_size * n_blocks, n_tokens, -1)

        for i in range(self.num_local_layers):
            word_vec = self.transformer_local[i](word_vec, word_vec, 1 - mask_local)

        mask_hier = mask_local[:, :, None].float()
        word_vec = word_vec * mask_hier
        word_vec = self.layer_norm1(word_vec)
        sent_vec = self.pooling(word_vec, mask_local)
        sent_vec = sent_vec.view(batch_size, n_blocks, -1)

        global_pos_emb = self.pos_emb.pe[:, :n_blocks]
        sent_vec = sent_vec+global_pos_emb

        sent_vec = self.layer_norm2(sent_vec)* mask_block.unsqueeze(-1).float()

        structure_vec = sent_vec
        roots = []
        for i in range(self.num_inter_layers):
            structure_vec, root = self.transformer_inter[i](sent_vec, structure_vec, 1 - mask_block)
            roots.append(root)



        return roots, mask_block



class TMTLayer(nn.Module):
    def __init__(self, d_model,  d_ff, dropout, iter):
        super(TMTLayer, self).__init__()

        self.iter = iter
        self.self_attn = StructuredAttention( d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
        self.linears1 = nn.ModuleList([nn.Linear(2*d_model,d_model) for _ in range(iter)])
        self.relu = nn.ReLU()
        self.linears2 = nn.ModuleList([nn.Linear(d_model,d_model) for _ in range(iter)])
        self.layer_norm = nn.ModuleList([nn.LayerNorm(d_model, eps=1e-6) for _ in range(iter)])

    def forward(self, x, structure_vec, mask):
        vecs = [x]

        mask = mask.unsqueeze(1)
        attn, root = self.self_attn(structure_vec,mask=mask)
        for i in range(self.iter):
            context = torch.matmul(attn, vecs[-1])
            new_c = self.linears2[i](self.relu(self.linears1[i](torch.cat([vecs[-1], context], -1))))
            new_c = self.layer_norm[i](new_c)
            vecs.append(new_c)

        return vecs[-1], root
