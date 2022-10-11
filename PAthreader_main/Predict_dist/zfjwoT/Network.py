import torch
import math
import torch.nn.functional as F
from torch import nn, einsum
import numpy as np


class ResidualNetwork(nn.Module):
    #                             511         64
    def __init__(self, n_feat_in, n_feat_out):
        super(ResidualNetwork, self).__init__()

        layer_s = list()
        layer_s.append(nn.InstanceNorm2d(n_feat_in, affine=True, eps=1e-6))
        layer_s.append(nn.ELU(inplace=False))
        layer_s.append(nn.Conv2d(n_feat_in, n_feat_out, 1, dilation=1, bias=False))

        self.layer = nn.Sequential(*layer_s)

    def forward(self, x):
        output = self.layer(x)
        return output


class FeedForwardLayer(nn.Module):
    def __init__(self, d_feat, d_ff, activation_dropout=0.1):
        super(FeedForwardLayer, self).__init__()
        self.d_feat = d_feat
        self.d_ff = d_ff
        self.activation_fn = nn.GELU()
        self.activation_dropout_module = nn.Dropout(activation_dropout, inplace=False)
        self.fc1 = nn.Linear(d_feat, d_ff)
        self.fc2 = nn.Linear(d_ff, d_feat)

    def forward(self, x):
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        return x


class TriangleMultiplicativeModule(nn.Module):
    def __init__(self, dim, orign_dim, mix='ingoing'):
        super(TriangleMultiplicativeModule, self).__init__()
        assert mix in {'ingoing', 'outgoing'}, 'mix must be either ingoing or outgoing'

        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, orign_dim)
        self.right_proj = nn.Linear(dim, orign_dim)

        self.left_gate = nn.Linear(dim, orign_dim)
        self.right_gate = nn.Linear(dim, orign_dim)
        self.out_gate = nn.Linear(dim, orign_dim)

        # initialize all gating to be identity

        for gate in (self.left_gate, self.right_gate, self.out_gate):
            nn.init.constant_(gate.weight, 0.)
            nn.init.constant_(gate.bias, 1.)

        if mix == 'outgoing':
            self.mix_einsum_eq = '... i k d, ... j k d -> ... i j d'
        elif mix == 'ingoing':
            self.mix_einsum_eq = '... k j d, ... k i d -> ... i j d'

        self.to_out_norm = nn.LayerNorm(orign_dim)
        self.to_out = nn.Linear(orign_dim, dim)

    def forward(self, x):
        assert x.shape[1] == x.shape[2], 'feature map must be symmetrical'

        x = self.norm(x)

        left = self.left_proj(x)
        right = self.right_proj(x)

        left_gate = self.left_gate(x).sigmoid()
        right_gate = self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum(self.mix_einsum_eq, left, right)

        out = self.to_out_norm(out)
        out = self.to_out(out)

        out = out * out_gate
        return out


class CoevolExtractor(nn.Module):
    #                     32           128
    def __init__(self, n_feat_proj, n_feat_out):
        super(CoevolExtractor, self).__init__()

        self.norm_2d = nn.LayerNorm(n_feat_proj * n_feat_proj)
        # project down to output dimension (pair feature dimension)
        self.proj_2 = nn.Linear(n_feat_proj ** 2, n_feat_out)

    def forward(self, x_down, x_down_w):
        B, N, L = x_down.shape[:3]

        pair = torch.einsum('abij,ablm->ailjm', x_down, x_down_w)  # outer-product & average pool # (1,L,L,32,32)
        pair = pair.reshape(B, L, L, -1)  # (1,L,L,1024)
        pair = self.norm_2d(pair)
        pair = self.proj_2(pair)  # (B, L, L, 128) # project down to pair dimension
        return pair


class SequenceWeight(nn.Module):
    #                    32       1
    def __init__(self, d_model, heads, dropout=0.15):
        super(SequenceWeight, self).__init__()
        self.heads = heads
        self.d_model = d_model
        self.d_k = d_model // heads
        self.scale = 1.0 / math.sqrt(self.d_k)

        self.to_query = nn.Linear(d_model, d_model)
        self.to_key = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout, inplace=False)

    def forward(self, msa):
        B, N, L = msa.shape[:3]

        msa = msa.permute(0, 2, 1, 3)  # (B, L, N, K)
        tar_seq = msa[:, :, 0].unsqueeze(2)  # (B, L, 1, K)

        q = self.to_query(tar_seq).view(B, L, 1, self.heads, self.d_k).permute(0, 1, 3, 2,
                                                                               4).contiguous()  # (B, L, h, 1, k)
        k = self.to_key(msa).view(B, L, N, self.heads, self.d_k).permute(0, 1, 3, 4, 2).contiguous()  # (B, L, h, k, N)

        q = q * self.scale
        attn = torch.matmul(q, k)  # (B, L, h, 1, N)
        attn = F.softmax(attn, dim=-1)
        return self.dropout(attn)


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
        self.ff = FeedForwardLayer(d_model, d_ff, activation_dropout=p_drop)

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


class AttentionBlock(nn.Module):
    def __init__(
            self,
            d_feat,
            p_drop,
            n_att_head,
            r_ff,
    ):
        super(AttentionBlock, self).__init__()

        self.dropout_module = nn.Dropout(p_drop, inplace=False)

        self.triangle_multiply_outgoing = TriangleMultiplicativeModule(dim=d_feat, orign_dim=d_feat, mix='outgoing')
        self.triangle_multiply_ingoing = TriangleMultiplicativeModule(dim=d_feat, orign_dim=d_feat, mix='ingoing')
        self.axial_attention = AxialEncoderLayer(d_model=d_feat, d_ff=d_feat * r_ff,
                                                 heads=n_att_head, p_drop=p_drop)

    def forward(self, x):
        x = self.dropout_module(self.triangle_multiply_outgoing(x)) + x
        x = self.dropout_module(self.triangle_multiply_ingoing(x)) + x
        x = self.axial_attention(x)
        return x


class Att(nn.Module):
    def __init__(self, n_layers, d_feat=64, n_att_head=8, p_drop=0.1, r_ff=4):
        super(Att, self).__init__()

        layer_s = list()
        for _ in range(n_layers):
            res_block = AttentionBlock(d_feat, p_drop, n_att_head, r_ff)
            layer_s.append(res_block)

        self.layer = nn.Sequential(*layer_s)

    def forward(self, x):
        output = self.layer(x)
        return output

class Conv2D(nn.Module):
    def __init__(self, n_c, out_channels=64, kernel=1, dilation=1):
        super(Conv2D, self).__init__()
        padding = self._get_same_padding(kernel, dilation)
        layer_s = list()
        layer_s.append(nn.InstanceNorm2d(n_c, affine=True, eps=1e-6))
        layer_s.append(nn.ELU(inplace=False))
        layer_s.append(nn.Conv2d(n_c, out_channels, kernel, padding=padding, dilation=dilation, bias=False))
        self.layer = nn.Sequential(*layer_s)

    def _get_same_padding(self, kernel, dilation):
        return (kernel + (kernel - 1) * (dilation - 1) - 1) // 2

    def forward(self, x):
        out = self.layer(x)
        return out


class subblock(nn.Module):
    def __init__(self, in_c, first=False, channels=64, dilations=1, expansion=4, scale=4, baseWidth=26):
        super(subblock, self).__init__()
        self.scale = scale
        self.expansion = expansion
        self.first = first
        self.width = int(baseWidth * channels / 64)  # 26,52

        if in_c == 1:
            if self.first:
                chan_in = 64
            else:
                chan_in = 256
        elif in_c == 2:
            if self.first:
                chan_in = 256
            else:
                chan_in = 512
        elif in_c == 3 or in_c == 4:
            chan_in = 512

        self.con1 = Conv2D(chan_in, out_channels=self.width * scale, kernel=1, dilation=1)
        self.con2 = Conv2D(self.width, out_channels=self.width, kernel=3, dilation=dilations)
        self.con3 = Conv2D(self.width, out_channels=self.width, kernel=3, dilation=dilations)
        self.con4 = Conv2D(self.width * scale, out_channels=channels * expansion, kernel=1)
        self.con5 = Conv2D(chan_in, out_channels=channels * expansion, kernel=1)

    def forward(self, input_x):
        outs = []
        x = self.con1(input_x)
        frac = np.int32(x.shape[1] / self.scale)
        for i in range(self.scale):
            if i == 0:
                outs.append(x[:, :frac, ...])
            elif i == 1:
                outs.append(self.con2(x[:, frac:2 * frac, ...]))
            elif i == 2:
                outs.append(self.con2(x[:, i * frac:(i + 1) * frac, ...]))
            else:
                xi = x[:, i * frac:(i + 1) * frac, ...] + outs[1]
                outs.append(self.con3(xi))
        out = torch.cat(outs, 1)
        out = self.con4(out)
        if self.first:
            input_x = self.con5(input_x)
        out += input_x

        return out


class block(nn.Module):
    def __init__(self, layers, in_c, d=1, out_channels=64):
        super(block, self).__init__()
        layer_s = list()

        sub1 = subblock(first=True, channels=out_channels, dilations=1, in_c=in_c)
        layer_s.append(sub1)
        for i in range(1, layers):
            d = 2 * d
            sub2 = subblock(channels=out_channels, dilations=d, in_c=in_c)
            layer_s.append(sub2)

        self.layer = nn.Sequential(*layer_s)

    def forward(self, input_x):
        out = self.layer(input_x)
        return out


class Res2Net(nn.Module):
    def __init__(self, layer=50):
        super(Res2Net, self).__init__()
        layers = []
        if layer == 50:
            layers = [3, 4, 6, 3]
        self.block1 = block(layers[0], in_c=1)  # [3, 4, 6, 3]
        self.block2 = block(layers[1], out_channels=128, in_c=2)
        self.block3 = block(layers[2], out_channels=128, in_c=3)
        self.block4 = block(layers[3], out_channels=128, in_c=4)

    def forward(self, input_x):
        out = self.block1(input_x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        return out


class Network(nn.Module):
    def __init__(self, d_features=447, d_msa=768, d_pair=128, n_feat_proj=32, p_drop=0.15,
                 n_attlayer=4):
        super(Network, self).__init__()
        # ==============================Msa_feat================================================
        # project down msa dimension (n_feat_in --> n_feat_proj)
        # 策略1
        self.norm = nn.LayerNorm(768)
        self.proj_1 = nn.Linear(d_msa, 256)
        self.norm_1 = nn.LayerNorm(256)
        self.proj_2 = nn.Linear(256, 64)
        self.norm_2 = nn.LayerNorm(64)
        self.proj_3 = nn.Linear(64, n_feat_proj)
        self.norm_3 = nn.LayerNorm(n_feat_proj)

        self.encoder = SequenceWeight(n_feat_proj, 1, dropout=p_drop)
        self.coevol = CoevolExtractor(n_feat_proj, d_pair)  # outer-product & average pool

        self.norm_new = nn.LayerNorm(d_pair)
        # ==============================Msa_feat================================================

        self.norm_47 = nn.LayerNorm(47)

        self.ResNet1 = ResidualNetwork(n_feat_in=d_features, n_feat_out=64)
        self.Attention = Att(n_layers=n_attlayer)
        self.res2Net = Res2Net()

        self.conv2d = nn.Conv2d(512, 37, 1)

    def forward(self, feat_47, msa, attention):
        # Input: MSA       (B,1,L,768)
        #        feat55    (B,L,L,55)
        #        att       (B,L,L,144)
        #        template  (B, T, L, L, 44)
        # Output: features (B,L,L,499)
        # =========================================process msa========================================
        B, N, L, _ = msa.shape
        # project down to reduce memory----------------------------------------
        # 策略1
        msa = self.norm(msa)
        msa = F.elu(self.norm_1(self.proj_1(msa)))
        msa = F.elu(self.norm_2(self.proj_2(msa)))
        msa_down = F.elu(self.norm_3(self.proj_3(msa)))  # down to(B,N,L,32)

        # ---------------------------------------------------------------------
        # 获取msa中每条序列的权重
        w_seq = self.encoder(msa_down).reshape(B, L, 1, N).permute(0, 3, 1, 2)  # (B,N,L,1)
        feat_1d = w_seq * msa_down  # (B,N,L,32)
        # 外积
        pair = self.coevol(msa_down, feat_1d)  # (1,L,L,128)
        pair = self.norm_new(pair)  # msa features

        # 乘以权重后的msa沿维度N聚合
        feat_1d = feat_1d.sum(1)  # (B,L,32)

        # 获取msa中查询序列
        query = msa_down[:, 0]  # (B,L,32)
        # additional 1D features
        feat_1d = torch.cat((feat_1d, query), dim=-1)  # (B,L,64)
        # tile 1D features
        left = feat_1d.unsqueeze(2).repeat(1, 1, L, 1)  # (1,L,L,64)
        right = feat_1d.unsqueeze(1).repeat(1, L, 1, 1)  # (1,L,L,64)

        # =========================================process msa========================================

        feat_47 = self.norm_47(feat_47)

        attention = 0.5 * (attention + attention.permute(0, 1, 3, 2))
        attention = attention.permute(0, 2, 3, 1).contiguous()

        # 所有特征连接在一起
        inputs = torch.cat((feat_47, pair, left, right, attention), -1)  # (1,L,L,256+144+47)
        inputs = inputs.permute(0, 3, 1, 2).contiguous()  # prep for convolution layer

        # ============================================net=============================================
        x = self.ResNet1(inputs)
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, L, L, C)
        x_att = self.Attention(x)
        x_att = x_att.permute(0, 3, 1, 2).contiguous()  # prep for convolution layer
        x_res = self.res2Net(x_att)
        x = 0.5 * (x_res + x_res.permute(0, 1, 3, 2))
        out = self.conv2d(x)
        return out
