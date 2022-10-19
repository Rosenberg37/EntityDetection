from torch import nn, Tensor

import modules
import modules.nn.detect
from modules import FormatSentence


class EntityDetection(nn.Module):
    def __init__(
            self,
            pretrain_select: str,
            model_max_length: int,
            pos_dim: int,
            char_dim: int,
            word2vec_select: str,
            use_backward: bool,
            use_category_embed: bool,
            use_logits_iter: bool,
            use_locations_iter: bool,
            use_spatial: bool,
            num_heads: int,
            num_layers: int,
            dim_feedforward: int,
            kernels_size: list[int],
            chars_list: list[str],
            pos_list: list[str],
            types2idx: dict,
            idx2types: dict,
            dropout: float,
    ):
        super(EntityDetection, self).__init__()
        embedding_kargs = {
            'pos_dim': pos_dim,
            'char_dim': char_dim,
            'word2vec_select': word2vec_select,
            'chars_list': chars_list,
            'pos_list': pos_list,
            'pretrain_select': pretrain_select,
            'model_max_length': model_max_length,
        }
        self.encoder = modules.Encoder(embedding_kargs)

        detector_kargs = {
            'hidden_size': self.encoder.hidden_size,
            'dropout': dropout,
            'types2idx': types2idx,
            'idx2types': idx2types,
            'proposer_kargs': {
                'use_backward': use_backward,
                'kernels_size': kernels_size
            },
            'regressor_kargs': {
                'num_layers': num_layers,
                'layer_kargs': {
                    'use_logits_iter': use_logits_iter,
                    'use_category_embed': use_category_embed,
                    'use_locations_iter': use_locations_iter,
                    'use_spatial': use_spatial,
                    'num_heads': num_heads,
                    'dim_feedforward': dim_feedforward,
                }
            },

        }
        self.detector = modules.Detector(**detector_kargs)

    def forward(self, batch_sentences: list[FormatSentence]) -> Tensor:
        """

        :param batch_sentences: (batch_size, sentence_length)
        :return: loss
        """
        batch_hiddens, batch_masks = self.encoder(batch_sentences)
        return self.detector(batch_hiddens, batch_masks, batch_sentences)

    def decode(self, batch_sentences: list[FormatSentence], return_detailed_results: bool = False) -> list[list[dict]]:
        """

        :param return_detailed_results: whether return the detail results.
        :param batch_sentences: (batch_size, sentence_length)
        :return: entities or more detailed results.
        """
        batch_hiddens, masks = self.encoder(batch_sentences)
        return self.detector.decode(batch_hiddens, masks, return_detailed_results)
