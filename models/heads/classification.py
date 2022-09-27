from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn.modules.transformer import _get_activation_fn


class ClassificationHead(nn.Module):
    def __init__(
        self,
        in_features,
        num_classes,
        reduction="flatten",
        dropout=None,
        return_logits=True,
    ):
        """
        Basic classification head for various tasks.

        Parameters
        ----------
        in_features: int
        num_classes: int
        reduction: nn.Module or str, default="flatten", optional
        dropout: float between (0.0, 1.0), default=None, optional
        return_logits: bool, default=True, optional
        """
        super(ClassificationHead, self).__init__()
        # build `fc` layer.
        self.fc = nn.Linear(in_features, num_classes)
        # build `reduction` layer.
        if isinstance(reduction, nn.Module):
            self.reduction = reduction
        elif type(reduction) == str:
            if reduction == "gap":
                self.reduction = nn.AdaptiveAvgPool2d((1, 1))
            elif reduction == "flatten":
                self.reduction = nn.Flatten()
            elif reduction == "none":
                self.reduction = nn.Identity()
            else:
                raise ValueError(f"Invalid value for `reduction`: {self.reduction}")
        # build dropout.
        if dropout:
            assert 0.0 <= dropout <= 1.0
            self.dropout = nn.Dropout(p=dropout)
        # build activation.
        if not return_logits:
            self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        # order: reduction -> dropout(optional) -> fc -> activation(optional).
        x = self.reduction(x)
        x = x.view(x.shape[0], -1)
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        x = self.fc(x)
        if hasattr(self, "activation"):
            x = self.activation(x)
        return x


"""
Implementation of the paper "ML-Decoder: Scalable and Versatile Classification Head"
Code from official implementation at:
    - https://github.com/Alibaba-MIIL/ML_Decoder/blob/main/src_files/ml_decoder/ml_decoder.py
"""


class _MLDecoderTransformerDecoderLayerOptimal(nn.Module):
    def __init__(
        self,
        d_model,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.0,
        activation="relu",
        layer_norm_eps=1e-5,
    ) -> None:
        super(_MLDecoderTransformerDecoderLayerOptimal, self).__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        assert dropout >= 0.0 and dropout <= 1.0
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = torch.nn.functional.relu
        super(_MLDecoderTransformerDecoderLayerOptimal, self).__setstate__(state)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        tgt = tgt + self.dropout1(tgt)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


# @torch.jit.script
# class ExtrapClasses(object):
#     def __init__(self, num_queries: int, group_size: int):
#         self.num_queries = num_queries
#         self.group_size = group_size
#
#     def __call__(self, h: torch.Tensor, class_embed_w: torch.Tensor, class_embed_b: torch.Tensor, out_extrap:
#     torch.Tensor):
#         # h = h.unsqueeze(-1).expand(-1, -1, -1, self.group_size)
#         h = h[..., None].repeat(1, 1, 1, self.group_size) # torch.Size([bs, 5, 768, groups])
#         w = class_embed_w.view((self.num_queries, h.shape[2], self.group_size))
#         out = (h * w).sum(dim=2) + class_embed_b
#         out = out.view((h.shape[0], self.group_size * self.num_queries))
#         return out


@torch.jit.script
class _MLDecoderGroupFC(object):
    def __init__(self, embed_len_decoder: int):
        self.embed_len_decoder = embed_len_decoder

    def __call__(
        self, h: torch.Tensor, duplicate_pooling: torch.Tensor, out_extrap: torch.Tensor
    ):
        for i in range(h.shape[1]):
            h_i = h[:, i, :]
            if len(duplicate_pooling.shape) == 3:
                w_i = duplicate_pooling[i, :, :]
            else:
                w_i = duplicate_pooling
            out_extrap[:, i, :] = torch.matmul(h_i, w_i)


class MLDecoderClassificationHead(nn.Module):
    def __init__(
        self,
        initial_num_features: int,
        num_classes: int,
        num_of_groups: int = None,
        dropout: float = 0.0,
        decoder_embedding: int = 768,
        zsl: bool = False,
        return_logits: bool = True,
    ):
        super(MLDecoderClassificationHead, self).__init__()
        embed_len_decoder = num_classes if num_of_groups is None else num_of_groups

        embed_standart = nn.Linear(initial_num_features, decoder_embedding)

        # non-learnable queries
        if not zsl:
            query_embed = nn.Embedding(embed_len_decoder, decoder_embedding)
            query_embed.requires_grad_(False)
        else:
            query_embed = None

        # decoder
        num_layers_decoder = 1
        dim_feedforward = 2048
        assert dropout >= 0.0 and dropout <= 1.0
        layer_decode = _MLDecoderTransformerDecoderLayerOptimal(
            d_model=decoder_embedding,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.decoder = nn.TransformerDecoder(
            layer_decode, num_layers=num_layers_decoder
        )
        self.decoder.embed_standart = embed_standart
        self.decoder.query_embed = query_embed
        self.zsl = zsl

        if self.zsl:
            if decoder_embedding != 300:
                self.wordvec_proj = nn.Linear(300, decoder_embedding)
            else:
                self.wordvec_proj = nn.Identity()
            self.decoder.duplicate_pooling = torch.nn.Parameter(
                torch.Tensor(decoder_embedding, 1)
            )
            self.decoder.duplicate_pooling_bias = torch.nn.Parameter(torch.Tensor(1))
            self.decoder.duplicate_factor = 1
        else:
            # group fully-connected
            self.decoder.num_classes = num_classes
            self.decoder.duplicate_factor = int(num_classes / embed_len_decoder + 0.999)
            self.decoder.duplicate_pooling = torch.nn.Parameter(
                torch.Tensor(
                    embed_len_decoder, decoder_embedding, self.decoder.duplicate_factor
                )
            )
            self.decoder.duplicate_pooling_bias = torch.nn.Parameter(
                torch.Tensor(num_classes)
            )
        torch.nn.init.xavier_normal_(self.decoder.duplicate_pooling)
        torch.nn.init.constant_(self.decoder.duplicate_pooling_bias, 0)
        self.decoder.group_fc = _MLDecoderGroupFC(embed_len_decoder)
        self.train_wordvecs = None
        self.test_wordvecs = None

        # build activation.
        if not return_logits:
            self.output_activation = nn.Softmax(dim=1)

    def forward(self, x):
        if len(x.shape) == 4:  # [bs,2048, 7,7]
            embedding_spatial = x.flatten(2).transpose(1, 2)
        else:  # [bs, 197,468]
            embedding_spatial = x
        embedding_spatial_786 = self.decoder.embed_standart(embedding_spatial)
        embedding_spatial_786 = torch.nn.functional.relu(
            embedding_spatial_786, inplace=True
        )

        bs = embedding_spatial_786.shape[0]
        if self.zsl:
            query_embed = torch.nn.functional.relu(
                self.wordvec_proj(self.decoder.query_embed)
            )
        else:
            query_embed = self.decoder.query_embed.weight
        # tgt = query_embed.unsqueeze(1).repeat(1, bs, 1)
        tgt = query_embed.unsqueeze(1).expand(
            -1, bs, -1
        )  # no allocation of memory with expand
        h = self.decoder(
            tgt, embedding_spatial_786.transpose(0, 1)
        )  # [embed_len_decoder, batch, 768]
        h = h.transpose(0, 1)

        out_extrap = torch.zeros(
            h.shape[0],
            h.shape[1],
            self.decoder.duplicate_factor,
            device=h.device,
            dtype=h.dtype,
        )
        self.decoder.group_fc(h, self.decoder.duplicate_pooling, out_extrap)
        if not self.zsl:
            h_out = out_extrap.flatten(1)[:, : self.decoder.num_classes]
        else:
            h_out = out_extrap.flatten(1)
        h_out += self.decoder.duplicate_pooling_bias
        logits = h_out

        if hasattr(self, "output_activation"):
            logits = self.output_activation(logits)
        return logits
