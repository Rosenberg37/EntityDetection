import logging
from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

import modules

logger = logging.getLogger(__name__)


class Proposer(nn.Module):
    def __init__(self, types_num: int, hidden_size: int, kernels_size: list[int], use_backward: bool, dropout: float):
        super().__init__()
        self.pyramid = Pyramid(hidden_size, kernels_size, use_backward, dropout)
        self.anchor_attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 2)
        )

        self.classifier = modules.Classifier(hidden_size, types_num)

        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch_hiddens: Tensor, batch_masks: Tensor) -> dict[str, Tensor]:
        """

        :param batch_masks:  [batch_size, sentence_length]
        :param batch_hiddens: [batch_size, sentence_length, hidden_size]
        :return: queries_kargs
            'queries': [batch_size, sentence_length, hidden_size]
            'queries_locs': [batch_size, sentence_length, 2]
            'queries_logits': [batch_size, sentence_length, types_num + 1]
        """
        batch_size, sentence_length = batch_hiddens.shape[:2]
        features, features_locs, features_masks = self.pyramid(batch_hiddens, batch_masks)

        features_locs, features = torch.cat(features_locs), torch.cat(features, dim=1)
        scores = self.anchor_attn(features).masked_fill_(~torch.cat(features_masks, dim=1).unsqueeze(-1), -1e12)

        indices = torch.arange(sentence_length, device=batch_hiddens.device).unsqueeze(-1)
        sel = indices.ge(features_locs[:, 0]) & indices.le(features_locs[:, 1])
        sel = torch.split(torch.nonzero(sel)[:, 1], torch.sum(sel, dim=-1).tolist())
        sel = pad_sequence(sel, padding_value=-1, batch_first=True)

        features, scores = features[:, sel], scores[:, sel]
        features_locs = features_locs[sel].unsqueeze(0).float()
        features_locs = torch.repeat_interleave(features_locs, dim=0, repeats=batch_size)
        scores.masked_fill_(sel.eq(-1).view(1, *sel.shape, 1), -1e12)

        weights = torch.softmax(scores, dim=-2)
        queries_locs = torch.einsum('blsc,blsc->blc', weights, features_locs)

        queries = torch.einsum('bls,blsh->blh', torch.mean(weights, dim=-1), features)
        queries = self.norm(batch_hiddens + self.dropout(queries))

        queries_logits = self.classifier(queries)

        return {
            'queries': queries,
            'queries_locs': queries_locs,
            'queries_logits': queries_logits,
        }


class Pyramid(nn.Module):
    def __init__(self, hidden_size: int, kernels_size: list, use_backward: bool, dropout: float):
        super().__init__()

        self.kernels_size = kernels_size
        self.visions_size = [1]
        for k in self.kernels_size:
            self.visions_size.append(self.visions_size[-1] + k - 1)

        self.forward_blocks = nn.ModuleList([ForwardBlock(hidden_size, dropout, k) for k in self.kernels_size + [None]])
        if use_backward:
            self.backward_blocks = nn.ModuleList(
                [BackwardBlock(hidden_size, dropout, *args)
                 for args in zip(self.visions_size, [None] + self.kernels_size)]
            )

    def forward(self, batch_hiddens: Tensor, batch_masks: Tensor) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        """

        :param batch_masks:  [batch_size, sentence_length]
        :param batch_hiddens: [batch_size, sentence_length, hidden_size]
        :return:
            features: (num_layers) list of [batch_size, length, hidden_size]
            locations: (num_layers) list of [length, 2]
            features_masks: (num_layers) list of [batch_size, length]
        """
        batch_size, sentence_length = batch_hiddens.shape[:2]
        indices = torch.arange(0, sentence_length, device=batch_hiddens.device)

        features = [batch_hiddens]
        features_locs, features_masks = list(), list()

        for i, (block, v_size) in enumerate(zip(self.forward_blocks, self.visions_size)):
            locs_s, locs_e = indices[:sentence_length - v_size + 1], indices[v_size - 1:]
            features_locs.append(torch.stack([locs_s, locs_e], dim=-1))
            features_masks.append(batch_masks[:, locs_e])

            features[i], next_features = block(features[i], features_masks[-1])
            features.append(next_features)

            if features[i] is next_features:
                break

        features, next_features = features[:-1], features[-1]
        if hasattr(self, 'backward_blocks'):
            for i, mask in enumerate(reversed(features_masks)):
                block = self.backward_blocks[len(features_masks) - i - 1]
                features[-(i + 1)], next_features = block(features[-(i + 1)], next_features, mask, batch_hiddens)

        return features, features_locs, features_masks


class ForwardBlock(nn.Module):
    def __init__(self, hidden_size: int, dropout: float, kernel_size: Optional[int] = None):
        super().__init__()
        self.kernel_size = kernel_size

        self.rnn_block = RNNBlock(hidden_size, dropout)
        if kernel_size is not None:
            self.conv_block = nn.Sequential(
                nn.Conv1d(hidden_size, hidden_size, (kernel_size,)),
                nn.GELU(),
            )

    def forward(self, features: Tensor, masks: Tensor) -> tuple[Tensor, Tensor]:
        """

        :param masks: [batch_size, sentence]
        :param features: [batch_size, features, hidden_size]
        :return: [batch_size, hidden_size, features - kernel_size + 1]
        """
        features = self.rnn_block(features, masks)
        if not hasattr(self, 'conv_block') or features.shape[1] < self.kernel_size:
            return features, features
        else:
            next_features = self.conv_block(features.transpose(-1, -2)).transpose(-1, -2)
            return features, next_features


class BackwardBlock(nn.Module):
    def __init__(self, hidden_size: int, dropout: float, vision_size: int, kernel_size: Optional[int] = None):
        super().__init__()
        self.rnn_block = RNNBlock(hidden_size, dropout)
        if kernel_size is not None:
            self.conv_block = nn.Sequential(
                nn.ConvTranspose1d(2 * hidden_size, hidden_size, (kernel_size,)),
                nn.GELU(),
            )
        self.trans_out = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Dropout(dropout)
        )
        self.pool = nn.MaxPool1d(vision_size, stride=1)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, features: Tensor, next_features: Tensor, masks: Tensor, batch_hiddens: Tensor) -> tuple[Tensor, Tensor]:
        """

        :param batch_hiddens: [batch_size, sentence_length, hidden_size]
        :param features: [batch_size, features_size, hidden_size]
        :param next_features: [batch_size, features, hidden_size]
        :param masks: [batch_size, sentence]
        :return: [batch_size, hidden_size, features - kernel_size + 1]
        """
        features = torch.cat([features, self.rnn_block(next_features, masks)], dim=-1)
        if hasattr(self, 'conv_block'):
            next_features = self.conv_block(features.transpose(-1, -2)).transpose(-1, -2)

        cut_features = self.pool(batch_hiddens.transpose(-1, -2)).transpose(-1, -2)
        features = self.norm(cut_features + self.trans_out(features))
        return features, next_features


class RNNBlock(nn.Module):
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.transform = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Dropout(dropout)
        )

    def forward(self, features: Tensor, masks: Tensor) -> Tensor:
        """

        :param features: [batch_size, features_size, hidden_size]
        :param masks: [batch_size, sentence]
        :return: [batch_size, features, hidden_size]
        """
        features = self.dropout(self.norm(features))
        lengths = torch.clamp_min(torch.sum(masks, dim=-1), 1).cpu()
        packed_features = pack_padded_sequence(features, lengths, batch_first=True, enforce_sorted=False)
        features = pad_packed_sequence(self.gru(packed_features)[0], batch_first=True)[0]
        return self.transform(features)
