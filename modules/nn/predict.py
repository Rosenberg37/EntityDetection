import math

import torch
from torch import nn, Tensor

from modules.nn import functional


class Predictor(nn.Module):
    def __init__(self, hidden_size: int, types_num: int):
        super().__init__()
        self.factor = math.sqrt(hidden_size)

        self.classifier = Classifier(hidden_size, types_num)
        self.trans_queries = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.Unflatten(-1, [2, hidden_size])
        )
        self.trans_context = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.Unflatten(-1, [2, hidden_size])
        )
        self.locator = Locator(hidden_size)

    def forward(self, batch_hiddens: Tensor, batch_masks: Tensor, queries: Tensor, queries_locs: Tensor, queries_logits: Tensor):
        """

        :param queries_logits: [batch_size, queries_size, types_num + 1]
        :param queries: [batch_size, queries_num, hidden_size]
        :param queries_locs: [batch_size, queries_num, 2]
        :param batch_hiddens: [batch_size, sentence_length, hidden_size]
        :param batch_masks: [batch_size, sentence_length]
        :return:
        """
        labels_pros = torch.softmax(queries_logits + self.classifier(queries), dim=-1)

        queries_h, context_h = self.trans_queries(queries), self.trans_context(batch_hiddens)

        score = torch.einsum('bqch,bvch->bcqv', queries_h, context_h) / self.factor
        score.masked_fill_(~batch_masks.unsqueeze(1).unsqueeze(1), -1e8)

        pairs = functional.get_table(torch.as_tensor([batch_hiddens.shape[1]], device=batch_hiddens.device))
        score = score[:, 0, :, pairs[:, 0]] + score[:, 1, :, pairs[:, 1]]

        mask, pairs = batch_masks[:, pairs[:, -1]], pairs.unsqueeze(0)
        offset, factors = self.locator(queries)
        weight = functional.distribute(pairs, mask, queries_locs + offset, factors)
        locs_pros = functional.weight_softmax(score, weight)

        return labels_pros, locs_pros


class Locator(nn.Module):
    def __init__(self, hidden_size: int, coordinate_dim: int = 2):
        super().__init__()
        self.coordinate_dim = coordinate_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, coordinate_dim * (coordinate_dim + 1))
        )

    def forward(self, queries: Tensor) -> tuple[Tensor, Tensor]:
        """

        :param queries: [..., hidden_size]
        :return:
            'centers': [..., 2(start&end)]
            'factors': [..., 2, 2]
        """
        splits = [self.coordinate_dim, self.coordinate_dim * self.coordinate_dim]
        offsets, factors = torch.split(self.mlp(queries), splits, dim=-1)
        factors = factors.view(*factors.shape[:-1], self.coordinate_dim, self.coordinate_dim)
        factors = factors @ factors.transpose(-1, -2)
        return offsets, factors


class Classifier(nn.Module):
    def __init__(self, hidden_size: int, types_num: int, additional_none: bool = True):
        super().__init__()
        self.labels_num = types_num
        if additional_none:
            self.none_label = types_num
            self.labels_num += 1

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, self.labels_num)
        )

    def forward(self, queries: Tensor) -> Tensor:
        """

        :param queries: [..., features_num, hidden_size]
        :return: [..., features_num, labels_num]
        """
        return self.mlp(queries)
