# Implements model described in the paper:
# Exploiting Temporal Contexts with Strided Transformer for 3D Human Pose Estimation
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

from einops import rearrange
from .mhformer import _TransformerEncoder


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, layer, N, length, d_model):
        super(Encoder, self).__init__()
        self.layers = layer
        self.norm = nn.LayerNorm(d_model)

        self.pos_embedding_1 = nn.Parameter(torch.randn(1, length, d_model))
        self.pos_embedding_2 = nn.Parameter(torch.randn(1, length, d_model))
        self.pos_embedding_3 = nn.Parameter(torch.randn(1, length, d_model))

    def forward(self, x, mask):
        for i, layer in enumerate(self.layers):
            if i == 0:
                x += self.pos_embedding_1[:, :x.shape[1]]
            elif i == 1:
                x += self.pos_embedding_2[:, :x.shape[1]]
            elif i == 2:
                x += self.pos_embedding_3[:, :x.shape[1]]

            x = layer(x, mask, i)

        return x


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout, stride_num, i):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.pooling = nn.MaxPool1d(1, stride_num[i])

    def forward(self, x, sublayer, i=-1, stride_num=-1):
        if i != -1:
            if stride_num[i] != 1:
                res = self.pooling(x.permute(0, 2, 1))
                res = res.permute(0, 2, 1)

                return res + self.dropout(sublayer(self.norm(x)))
            else:
                return x + self.dropout(sublayer(self.norm(x)))
        else:
            return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout, stride_num, i):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.stride_num = stride_num
        self.sublayer = clones(SublayerConnection(size, dropout, stride_num, i), 2)
        self.size = size

    def forward(self, x, mask, i):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.sublayer[1](x, self.feed_forward, i, self.stride_num)
        return x


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h 
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, number=-1, stride_num=-1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Conv1d(d_model, d_ff, kernel_size=1, stride=1)
        self.w_2 = nn.Conv1d(d_ff, d_model, kernel_size=3, stride=stride_num[number], padding=1)

        self.gelu = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.w_2(self.dropout(self.gelu(self.w_1(x))))
        x = x.permute(0, 2, 1)

        return x


class _TransformerReduce(nn.Module):
    def __init__(self, n_layers=3, d_model=256, d_ff=512, h=8, length=27, stride_num=None, dropout=0.1):
        super().__init__()

        self.length = length

        self.stride_num = stride_num
        self.model = self.make_model(N=n_layers, d_model=d_model, d_ff=d_ff, h=h, dropout=dropout, length = self.length)

    def forward(self, x, mask=None):
        x = self.model(x, mask)

        return x

    def make_model(self, N=3, d_model=256, d_ff=512, h=8, dropout=0.1, length=27):
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)

        model_EncoderLayer = []
        for i in range(N):
            ff = PositionwiseFeedForward(d_model, d_ff, dropout, i, self.stride_num)
            model_EncoderLayer.append(EncoderLayer(d_model, c(attn), c(ff), dropout, self.stride_num, i))

        model_EncoderLayer = nn.ModuleList(model_EncoderLayer)

        model = Encoder(model_EncoderLayer, N, length, d_model)

        return model


class StridedTransformer(nn.Module):
    def __init__(self, num_frame=351, num_joints=17, layers=3, strides=[3, 9, 13], channel=512, d_hid=512):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(2 * num_joints, channel, kernel_size=1),
            nn.BatchNorm1d(channel, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25)
        )

        self.Transformer = _TransformerEncoder(
            depth=layers, embed_dim=channel, mlp_hidden_dim=d_hid, length=num_frame)
        self.Transformer_reduce = _TransformerReduce(len(strides), channel, d_hid,
                                                     length=num_frame, stride_num=strides)

        self.fcn = nn.Sequential(
            nn.BatchNorm1d(channel, momentum=0.1),
            nn.Conv1d(channel, 3 * num_joints, kernel_size=1)
        )

        # self.fcn_1 = nn.Sequential(
        #     nn.BatchNorm1d(channel, momentum=0.1),
        #     nn.Conv1d(channel, 3 * num_joints, kernel_size=1)
        # )

    def forward(self, x):
        B, J, C, F = x.shape
        x = rearrange(x, 'b j c f -> b (j c) f').contiguous()

        x = self.encoder(x)
        x = x.permute(0, 2, 1).contiguous()

        x = self.Transformer(x)

        # x_VTE = x
        # x_VTE = x_VTE.permute(0, 2, 1).contiguous()
        # x_VTE = self.fcn_1(x_VTE)
        # x_VTE = rearrange(x_VTE, 'b (j c) f -> b f j c', j=J).contiguous()

        x = self.Transformer_reduce(x)

        x = x.permute(0, 2, 1).contiguous()
        x = self.fcn(x)
        x = rearrange(x, 'b (j c) f -> b f j c', j=J).contiguous()

        return x  # , x_VTE
