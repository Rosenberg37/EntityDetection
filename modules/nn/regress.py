import math
from typing import Union

import torch
from torch import nn, Tensor

import modules


class Regressor(nn.Module):
    def __init__(self, num_layers: int, layer_kargs: dict):
        super(Regressor, self).__init__()
        self.layers = nn.ModuleList([IterativeLayer(**layer_kargs) for _ in range(num_layers)])

    def forward(
            self,
            queries_kargs: dict[str, Tensor],
            batch_masks: Tensor,
            return_layers: bool = False
    ) -> Union[list[dict[str, Tensor]], dict[str, Tensor]]:
        """

        :param return_layers: whether return each layers' results.
        :param batch_masks: [batch_size, sentence_length]
        :param queries_kargs:
            'queries': [batch_size, sentence_length, hidden_size]
            'locations': [batch_size, sentence_length, 2]
            'logits': [batch_size, sentence_length, types_num + 1]
        :return: queries_kargs:
            'queries': [batch_size, sentence_length, hidden_size]
            'locations': [batch_size, sentence_length, 2]
            'logits': [batch_size, sentence_length, types_num + 1]
        """
        if not return_layers:
            for layer in self.layers:
                queries_kargs.update(layer(batch_masks=batch_masks, **queries_kargs))
            return queries_kargs
        else:
            layers_kargs = [queries_kargs]
            for layer in self.layers:
                layers_kargs.append(layer(batch_masks=batch_masks, **layers_kargs[-1]))
            return layers_kargs


class IterativeLayer(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            types_num: int,
            dim_feedforward: int,
            use_category_embed: bool,
            use_locations_iter: bool,
            use_logits_iter: bool,
            use_spatial: bool,
            dropout: float
    ):
        super(IterativeLayer, self).__init__()
        if use_category_embed:
            self.types_embeds = nn.Parameter(torch.empty(types_num + 1, hidden_size))
            torch.nn.init.normal_(self.types_embeds)

        self.attention_block = AttentionBlock(hidden_size, num_heads, use_locations_iter, use_spatial, dropout)
        self.linear_block = LinearBlock(hidden_size, dim_feedforward, dropout)

        if use_logits_iter:
            self.classifier = modules.Classifier(hidden_size, types_num)

    def forward(self, queries: Tensor, queries_locs: Tensor, queries_logits: Tensor, batch_masks: Tensor) -> dict[str, Tensor]:
        """


        :param queries: [batch_size, queries_num, hidden_size]
        :param queries_locs: [batch_size, queries_num, 2]
        :param queries_logits: [batch_size, queries_num, types_num + 1]
        :param batch_masks: [batch_size, queries_num]
        :return: queries_kargs:
            'queries': [batch_size, sentence_length, hidden_size]
            'locations': [batch_size, sentence_length, 2]
            'logits': [batch_size, sentence_length, types_num + 1]
        """
        if hasattr(self, 'types_embeds'):
            queries = queries + torch.matmul(torch.softmax(queries_logits, dim=-1), self.types_embeds)
        queries, queries_locs = self.attention_block(queries, queries_locs, batch_masks)
        queries = self.linear_block(queries)
        if hasattr(self, 'classifier'):
            queries_logits = queries_logits + self.classifier(queries)
        return {
            'queries': queries,
            'queries_locs': queries_locs,
            'queries_logits': queries_logits,
        }


class AttentionBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, use_locations_iter: bool, use_spatial: bool, dropout: float):
        super(AttentionBlock, self).__init__()
        self.use_locations_iter = use_locations_iter
        self.use_spatial = use_spatial

        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.factor = math.sqrt(head_dim)

        self.transform = nn.Sequential(
            nn.Linear(hidden_size, 3 * num_heads * head_dim),
            nn.Unflatten(-1, [3 * num_heads, head_dim])
        )

        self.attn_dropout = nn.Dropout(dropout)

        self.heads_attn = nn.Sequential(
            nn.Linear(head_dim, head_dim // 2),
            nn.GELU(),
            nn.Linear(head_dim // 2, 2)
        )

        self.locator = modules.Locator(head_dim)

        self.norm = nn.LayerNorm(hidden_size)
        self.trans_out = nn.Sequential(
            nn.Linear(num_heads * head_dim, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, queries: Tensor, locations: Tensor, masks: Tensor) -> tuple[Tensor, Tensor]:
        """


        :param queries: [batch_size, queries_num, hidden_size]
        :param locations: [batch_size, queries_num, 2]
        :param masks: [batch_size, queries_num]
        :return:
            queries: [batch_size, queries_size, hidden_size]
            locations: [batch_size, queries_num, 2]
        """

        queries_h, keys_h, values_h = torch.chunk(self.transform(queries).transpose(1, 2), 3, dim=1)
        score = torch.einsum('bnqh,bnvh->bnqv', queries_h, keys_h) / self.factor
        score.masked_fill_(~masks.unsqueeze(1).unsqueeze(1), -1e12)

        offsets, factors = self.locator(queries_h)
        locs = locations.unsqueeze(1) + offsets

        if self.use_spatial:
            spatial_weight = modules.distribute(locations.unsqueeze(1), masks.unsqueeze(1), locs, factors)
            attn_weight = self.attn_dropout(modules.weight_softmax(score, spatial_weight))
        else:
            attn_weight = self.attn_dropout(torch.softmax(score, dim=-1))

        outputs_h = torch.einsum('bnqv,bnvh->bqnh', attn_weight, values_h)
        heads_weights = torch.softmax(self.heads_attn(outputs_h).squeeze(-1), dim=-2)

        if self.use_locations_iter:
            locations = torch.einsum('bnqc,bqnc->bqc', locs, heads_weights)
        heads_weights = torch.mean(heads_weights, dim=-1)

        outputs_h = torch.einsum('bqnh,bqn->bqnh', outputs_h, heads_weights)
        queries = self.norm(self.trans_out(outputs_h.flatten(-2)) + queries)

        return queries, locations


class LinearBlock(nn.Module):
    def __init__(self, hidden_size: int, dim_feedforward: int, dropout: float):
        super(LinearBlock, self).__init__()
        self.transform = nn.Sequential(
            nn.Linear(hidden_size, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_size),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, queries: Tensor) -> tuple[Tensor, Tensor]:
        """

        :param queries: [batch_size, queries_num, hidden_size]
        :return: [batch_size, queries_num, hidden_size]
        """
        return self.norm(queries + self.transform(queries))
