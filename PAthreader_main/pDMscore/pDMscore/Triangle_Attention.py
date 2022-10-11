import torch
import torch.nn.functional as F
import math
import copy
from torch import nn, einsum


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + self.eps)
        x = self.a_2 * (x - mean)
        x /= std
        x += self.b_2
        return x


class FeedForwardLayer(nn.Module):
    def __init__(self, d_model, d_ff, p_drop=0.1):
        super(FeedForwardLayer, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.constant_(self.linear1.bias, 0)

        self.dropout = nn.Dropout(p_drop, inplace=False)
        self.linear2 = nn.Linear(d_ff, d_model)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.constant_(self.linear2.bias, 0)

    def forward(self, src):
        src = self.linear2(self.dropout(F.relu_(self.linear1(src))))
        return src


class Encoder(nn.Module):
    def __init__(self, enc_layer, n_layer):
        super(Encoder, self).__init__()
        self.layers = _get_clones(enc_layer, n_layer)
        self.n_layer = n_layer

    def forward(self, src, return_att=False):
        output = src
        for layer in self.layers:
            output = layer(output, return_att=return_att)
        return output


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, heads, k_dim=None, v_dim=None, dropout=0.1):
        super(MultiheadAttention, self).__init__()
        if k_dim == None:
            k_dim = d_model
        if v_dim == None:
            v_dim = d_model

        self.heads = heads
        self.d_model = d_model
        self.d_k = d_model // heads
        self.scaling = 1 / math.sqrt(self.d_k)

        self.to_query = nn.Linear(d_model, d_model)
        self.to_key = nn.Linear(k_dim, d_model)
        self.to_value = nn.Linear(v_dim, d_model)
        self.to_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout, inplace=False)

    def forward(self, query, key, value, return_att=False):
        batch, L1 = query.shape[:2]
        batch, L2 = key.shape[:2]
        q = self.to_query(query).view(batch, L1, self.heads, self.d_k).permute(0, 2, 1, 3)  # (B, h, L, d_k)
        k = self.to_key(key).view(batch, L2, self.heads, self.d_k).permute(0, 2, 1, 3)  # (B, h, L, d_k)
        v = self.to_value(value).view(batch, L2, self.heads, self.d_k).permute(0, 2, 1, 3)
        #
        attention = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        attention = F.softmax(attention, dim=-1)  # (B, h, L1, L2)
        attention = self.dropout(attention)
        #
        out = torch.matmul(attention, v)  # (B, h, L, d_k)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch, L1, -1)
        #
        out = self.to_out(out)
        if return_att:
            attention = 0.5 * (attention + attention.permute(0, 1, 3, 2))
            return out, attention.permute(0, 2, 3, 1)
        return out


class AxialEncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, heads, p_drop=0.1):
        super(AxialEncoderLayer, self).__init__()

        # multihead attention
        self.attn_L = MultiheadAttention(d_model, heads, dropout=p_drop)
        self.attn_N = MultiheadAttention(d_model, heads, dropout=p_drop)

        # feedforward
        self.ff = FeedForwardLayer(d_model, d_ff, p_drop=p_drop)

        # normalization module
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p_drop, inplace=False)
        self.dropout2 = nn.Dropout(p_drop, inplace=False)
        self.dropout3 = nn.Dropout(p_drop, inplace=False)

    def forward(self, src, return_att=False):
        # Input shape for multihead attention: (BATCH, L, L, EMB)
        B, N, L = src.shape[:3]

        # attention over L
        src2 = self.norm1(src)
        src2 = src2.reshape(B * N, L, -1)
        src2 = self.attn_L(src2, src2, src2)
        src2 = src2.reshape(B, N, L, -1)
        src = src + self.dropout1(src2)

        # attention over N
        src2 = self.norm2(src)
        src2 = src2.permute(0, 2, 1, 3).reshape(B * L, N, -1)
        src2 = self.attn_N(src2, src2, src2)  # attention over N
        src2 = src2.reshape(B, L, N, -1).permute(0, 2, 1, 3)
        src = src + self.dropout2(src2)

        # feed-forward
        src2 = self.norm3(src)  # pre-normalization
        src2 = self.ff(src2)
        src = src + self.dropout3(src2)
        return src


class AttentionModule(nn.Module):
    def __init__(self, n_layer, n_att_head=8, n_feat=128, r_ff=4, p_drop=0.1):
        super(AttentionModule, self).__init__()
        enc_layer = AxialEncoderLayer(d_model=n_feat, d_ff=n_feat * r_ff,
                                      heads=n_att_head, p_drop=p_drop)

        self.encoder = Encoder(enc_layer, n_layer)

    def forward(self, x):
        return self.encoder(x)
