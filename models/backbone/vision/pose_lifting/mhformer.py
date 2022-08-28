# Implements model described in the paper:
# MHFormer: Multi-Hypothesis Transformer for 3D Human Pose Estimation, CVPR 2022
import torch
import torch.nn as nn
from einops import rearrange
from functools import partial
from timm.models.layers import DropPath
from .poseformer import Block, Mlp, Attention


########################################################################
# Transformer encoder(vanilla transformer)
# reference: https://github.com/Vegetebird/MHFormer/blob/fcf238631016f906477ec9c1d17582097ecf9803/
# model/module/trans.py
########################################################################
class _TransformerEncoder(nn.Module):
    def __init__(self, depth=3, embed_dim=512, mlp_hidden_dim=1024, h=8, drop_rate=0.1, length=27):
        super().__init__()
        drop_path_rate = 0.2
        attn_drop_rate = 0.
        qkv_bias = True
        qk_scale = None

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.pos_embed = nn.Parameter(torch.zeros(1, length, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=h, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        x += self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        return x


########################################################################
# Hypothesis Transformer
# reference: https://github.com/Vegetebird/MHFormer/blob/fcf238631016f906477ec9c1d17582097ecf9803/
# model/module/trans_hypothesis.py
########################################################################
class _CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.linear_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear_v = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x_1, x_2, x_3):
        B, N, C = x_1.shape
        q = self.linear_q(x_1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.linear_k(x_2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.linear_v(x_3).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class _SHR_Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_dim, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1_1 = norm_layer(dim)
        self.norm1_2 = norm_layer(dim)
        self.norm1_3 = norm_layer(dim)

        self.attn_1 = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.attn_2 = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.attn_3 = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim * 3)
        self.mlp = Mlp(in_features=dim * 3, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x_1, x_2, x_3):
        x_1 = x_1 + self.drop_path(self.attn_1(self.norm1_1(x_1)))
        x_2 = x_2 + self.drop_path(self.attn_2(self.norm1_2(x_2)))
        x_3 = x_3 + self.drop_path(self.attn_3(self.norm1_3(x_3)))

        x = torch.cat([x_1, x_2, x_3], dim=2)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x_1 = x[:, :, :x.shape[2] // 3]
        x_2 = x[:, :, x.shape[2] // 3: x.shape[2] // 3 * 2]
        x_3 = x[:, :, x.shape[2] // 3 * 2: x.shape[2]]

        return x_1, x_2, x_3


class _CHI_Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_dim, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm3_11 = norm_layer(dim)
        self.norm3_12 = norm_layer(dim)
        self.norm3_13 = norm_layer(dim)

        self.norm3_21 = norm_layer(dim)
        self.norm3_22 = norm_layer(dim)
        self.norm3_23 = norm_layer(dim)

        self.norm3_31 = norm_layer(dim)
        self.norm3_32 = norm_layer(dim)
        self.norm3_33 = norm_layer(dim)

        self.attn_1 = _CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                      qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.attn_2 = _CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                      qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.attn_3 = _CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                      qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim * 3)
        self.mlp = Mlp(in_features=dim * 3, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x_1, x_2, x_3):
        x_1 = x_1 + self.drop_path(self.attn_1(self.norm3_11(x_2), self.norm3_12(x_3), self.norm3_13(x_1)))    
        x_2 = x_2 + self.drop_path(self.attn_2(self.norm3_21(x_1), self.norm3_22(x_3), self.norm3_23(x_2)))  
        x_3 = x_3 + self.drop_path(self.attn_3(self.norm3_31(x_1), self.norm3_32(x_2), self.norm3_33(x_3)))  

        x = torch.cat([x_1, x_2, x_3], dim=2)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x_1 = x[:, :, :x.shape[2] // 3]
        x_2 = x[:, :, x.shape[2] // 3: x.shape[2] // 3 * 2]
        x_3 = x[:, :, x.shape[2] // 3 * 2: x.shape[2]]

        return x_1, x_2, x_3


class _TransformerHypothesis(nn.Module):
    def __init__(self, depth=3, embed_dim=512, mlp_hidden_dim=1024, h=8, drop_rate=0.1, length=27):
        super().__init__()
        drop_path_rate = 0.20
        attn_drop_rate = 0.
        qkv_bias = True
        qk_scale = None

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.pos_embed_1 = nn.Parameter(torch.zeros(1, length, embed_dim))
        self.pos_embed_2 = nn.Parameter(torch.zeros(1, length, embed_dim))
        self.pos_embed_3 = nn.Parameter(torch.zeros(1, length, embed_dim))

        self.pos_drop_1 = nn.Dropout(p=drop_rate)
        self.pos_drop_2 = nn.Dropout(p=drop_rate)
        self.pos_drop_3 = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  

        self.SHR_blocks = nn.ModuleList([
            _SHR_Block(
                dim=embed_dim, num_heads=h, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth - 1)])

        self.CHI_blocks = nn.ModuleList([
            _CHI_Block(
                dim=embed_dim, num_heads=h, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[depth - 1], norm_layer=norm_layer)
            for i in range(1)])

        self.norm = norm_layer(embed_dim * 3)

    def forward(self, x_1, x_2, x_3):
        x_1 += self.pos_embed_1
        x_2 += self.pos_embed_2
        x_3 += self.pos_embed_3

        x_1 = self.pos_drop_1(x_1)
        x_2 = self.pos_drop_2(x_2)
        x_3 = self.pos_drop_3(x_3)

        for i, blk in enumerate(self.SHR_blocks):
            x_1, x_2, x_3 = self.SHR_blocks[i](x_1, x_2, x_3)

        x_1, x_2, x_3 = self.CHI_blocks[0](x_1, x_2, x_3)

        x = torch.cat([x_1, x_2, x_3], dim=2)
        x = self.norm(x)

        return x


########################################################################
# Combined MHFormer model
# reference: https://github.com/Vegetebird/MHFormer/blob/fcf238631016f906477ec9c1d17582097ecf9803/
# model/mhformer.py
########################################################################
class MHFormer(nn.Module):
    def __init__(self, num_frame=81, num_joints=17, layers=3, channel=512, d_hid=1024):
        super().__init__()

        # MHG
        self.norm_1 = nn.LayerNorm(num_frame)
        self.norm_2 = nn.LayerNorm(num_frame)
        self.norm_3 = nn.LayerNorm(num_frame)

        self.Transformer_encoder_1 = _TransformerEncoder(4, num_frame, num_frame * 2, length=2 * num_joints, h=9)
        self.Transformer_encoder_2 = _TransformerEncoder(4, num_frame, num_frame * 2, length=2 * num_joints, h=9)
        self.Transformer_encoder_3 = _TransformerEncoder(4, num_frame, num_frame * 2, length=2 * num_joints, h=9)

        # Embedding
        if num_frame > 27:
            self.embedding_1 = nn.Conv1d(2 * num_joints, channel, kernel_size=1)
            self.embedding_2 = nn.Conv1d(2 * num_joints, channel, kernel_size=1)
            self.embedding_3 = nn.Conv1d(2 * num_joints, channel, kernel_size=1)
        else:
            self.embedding_1 = nn.Sequential(
                nn.Conv1d(2 * num_joints, channel, kernel_size=1),
                nn.BatchNorm1d(channel, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )

            self.embedding_2 = nn.Sequential(
                nn.Conv1d(2 * num_joints, channel, kernel_size=1),
                nn.BatchNorm1d(channel, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )

            self.embedding_3 = nn.Sequential(
                nn.Conv1d(2 * num_joints, channel, kernel_size=1),
                nn.BatchNorm1d(channel, momentum=0.1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.25)
            )

        # SHR & CHI
        self.Transformer_hypothesis = _TransformerHypothesis(layers, channel, d_hid, length=num_frame)

        # Regression
        self.regression = nn.Sequential(
            nn.BatchNorm1d(channel * 3, momentum=0.1),
            nn.Conv1d(channel * 3, 3 * num_joints, kernel_size=1)
        )

    def forward(self, x):
        B, J, C, F = x.shape
        x = rearrange(x, 'b j c f -> b (j c) f').contiguous()

        # MHG
        x_1 = x + self.Transformer_encoder_1(self.norm_1(x))
        x_2 = x_1 + self.Transformer_encoder_2(self.norm_2(x_1))
        x_3 = x_2 + self.Transformer_encoder_3(self.norm_3(x_2))

        # Embedding
        x_1 = self.embedding_1(x_1).permute(0, 2, 1).contiguous()
        x_2 = self.embedding_2(x_2).permute(0, 2, 1).contiguous()
        x_3 = self.embedding_3(x_3).permute(0, 2, 1).contiguous()

        # SHR & CHI
        x = self.Transformer_hypothesis(x_1, x_2, x_3)

        # Regression
        x = x.permute(0, 2, 1).contiguous()
        x = self.regression(x)
        x = rearrange(x, 'b (j c) f -> b f j c', j=J).contiguous()

        return x
