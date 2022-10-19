from functools import reduce
from typing import Optional, Union

import torch
from scipy.optimize import linear_sum_assignment
from torch import nn, Tensor
from torch.types import Device

import modules
from modules import FormatSentence


class Detector(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            dropout: float,
            proposer_kargs: dict,
            regressor_kargs: dict,
            types2idx: dict,
            idx2types: dict
    ):
        super().__init__()
        types_num = len(types2idx)
        self.types2idx, self.idx2types = types2idx, idx2types

        general_args = {
            'types_num': types_num,
            'hidden_size': hidden_size,
            'dropout': dropout,
        }

        proposer_kargs.update(general_args)
        self.proposer = modules.Proposer(**proposer_kargs)

        regressor_kargs['layer_kargs'].update(general_args)
        self.regressor = modules.Regressor(**regressor_kargs)

        self.predictor = modules.Predictor(hidden_size, types_num)

    def predict(
            self,
            batch_hiddens: Tensor,
            batch_masks: Tensor,
            return_detailed_results: bool = False
    ) -> tuple[tuple[Tensor, Tensor], Optional[list[dict[str, Tensor]]]]:
        if not return_detailed_results:
            queries_kargs = self.proposer(batch_hiddens, batch_masks)
            queries_kargs = self.regressor(queries_kargs, batch_masks)
            return self.predictor(batch_hiddens, batch_masks, **queries_kargs), None
        else:
            iterated_kargs = [self.proposer(batch_hiddens, batch_masks)]
            iterated_kargs.extend(self.regressor(iterated_kargs[-1], batch_masks, True))
            proposals = list(map(lambda kargs: {
                'queries_locs': kargs['queries_locs'],
                'queries_logits': torch.softmax(kargs['queries_logits'], dim=-1),
            }, iterated_kargs))
            return self.predictor(batch_hiddens, batch_masks, **iterated_kargs[-1]), proposals

    def forward(self, batch_hiddens: Tensor, batch_masks: Tensor, batch_sentences: list[FormatSentence]) -> Tensor:
        """

        :param batch_hiddens: [batch_size, sentence_length, hidden_size]
        :param batch_masks: [batch_size, sentence_length]
        :param batch_sentences: (batch_size) list of sentence containing entities
        :return: loss
        """
        batch_targets = self.format(batch_sentences, batch_hiddens.device)
        (labels_pros, locs_pros), _ = self.predict(batch_hiddens, batch_masks)

        losses = list()
        for labels_p, locs_p, masks, targets in zip(labels_pros, locs_pros, batch_masks, batch_targets):
            entity_p = locs_p[masks][:, modules.flatten(targets[:, :2])] * labels_p[masks][:, targets[:, 2]]
            cost = -torch.log(torch.clamp_min(entity_p, 1e-8))
            row_idx, col_inx = linear_sum_assignment(cost.cpu().detach().numpy())
            neg_p = labels_p[list(set(range(torch.sum(masks))) - set(row_idx)), -1]
            loss = torch.cat([cost[row_idx, col_inx], -torch.log(torch.clamp_min(neg_p, 1e-8))])
            losses.append(torch.mean(loss))
        return sum(losses) / len(losses)

    def decode(
            self,
            batch_hiddens: Tensor,
            batch_masks: Tensor,
            return_detailed_results: bool = False
    ) -> Union[list[list], tuple[list[list], list[dict[str, Tensor]]]]:
        """

        :param return_detailed_results: whether return the detail results.
        :param batch_hiddens: [batch_size, sentence_length, hidden_size]
        :param batch_masks: [batch_size, sentence_length]
        :return: (batch_size, entities_num, 3)
            0: start_idx
            1: end_idx
            2: type
        """
        if return_detailed_results and batch_hiddens.shape[0] > 1:
            raise RuntimeError("Detail result only support batch_size = 1")

        length = torch.as_tensor([batch_hiddens.shape[1]]).to(batch_hiddens.device)
        (labels_pros, locs_pros), results = self.predict(batch_hiddens, batch_masks, return_detailed_results)

        batch_triples = list()
        for labels_p, locs_p, mask in zip(labels_pros, locs_pros, batch_masks):
            locs_p, labels_p = locs_p[mask], labels_p[mask]
            locs_p, loc = torch.max(locs_p, dim=-1)
            types_p, types = torch.max(labels_p[:, :-1], dim=-1)
            sel = (locs_p * types_p).gt(labels_p[:, -1])
            triples = torch.cat([modules.relocate(loc, length), types.unsqueeze(-1)], dim=-1)
            batch_triples.append(triples[sel].tolist())

            if return_detailed_results:
                for result in results:
                    for key, value in result.items():
                        result[key] = value[0, sel]

        batch_entities = self.inverse_format(batch_triples)

        if return_detailed_results:
            return batch_entities, results
        else:
            return batch_entities

    def format(self, batch_sentences: list[FormatSentence], device: Device) -> list[Tensor]:
        """

        :param device: device tensor created on
        :param batch_sentences: (batch_size, entities_num) of dicts
             'start_idx': start index of entity
             'end_idx': end index of entity
             'type': type of entity
        :return: (batch_size) list of [entities_num, 3]
        """
        batch_targets = list()
        for sentence in batch_sentences:
            entities = sentence.entities
            if len(entities) == 0:
                targets = torch.zeros([0, 3], device=device, dtype=torch.long)
            else:
                targets = [[entity['start'], entity['end'] - 1, self.types2idx[entity['type']]] for entity in entities]
                targets = torch.as_tensor(targets, device=device, dtype=torch.long)
            batch_targets.append(targets)
        return batch_targets

    def inverse_format(self, batch_triples: list[list]) -> list[list[dict]]:
        """

        :param batch_triples: (batch_size, entities_num, 3)
        :return: batch_entities: (batch_size, entities_num) of dicts
        """
        batch_entities = list()
        for i, triples in enumerate(batch_triples):
            batch_entities.append([{
                'start': triple[0],
                'end': triple[1] + 1,
                'type': self.idx2types[triple[2]]
            } for triple in triples])
        return list(map(lambda a: reduce(lambda x, y: x if y in x else x + [y], [[], *a]), batch_entities))
