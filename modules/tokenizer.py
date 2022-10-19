import logging
from typing import Union

import transformers
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


def build_tokenizer(pretrain_select: str, cache_dir: str):
    return Tokenizer({'return_tensors': 'pt', 'padding': True},
                     transformers.AutoTokenizer.from_pretrained(pretrain_select, cache_dir=cache_dir, padding_side='right'))


class Tokenizer:
    accents_table = str.maketrans(
        "áéíóúýàèìòùỳâêîôûŷäëïöüÿñÁÉÍÓÚÝÀÈÌÒÙỲÂÊÎÔÛŶÄËÏÖÜŸ",
        "aeiouyaeiouyaeiouyaeiouynAEIOUYAEIOUYAEIOUYAEIOUY"
    )
    special_tokens = {
        '“': '"',
        '”': '"',
        '‘': ''',
        '’': ''',
        '—': '-',
        '…': '...',
        '……': '...',
        '�': '?'
    }

    def __init__(self, encode_kargs: dict, tokenizer: PreTrainedTokenizerFast = None):
        self.encode_kargs = encode_kargs
        self.tokenizer = tokenizer

    def __call__(self, batch_texts: Union[list[str], list[list[str]]]):
        batch_texts = list(map(self.preprocess, batch_texts))
        return self.tokenizer(batch_texts, is_split_into_words=True, **self.encode_kargs)

    def batch_decode(self, *args, **kargs):
        batch_texts = self.tokenizer.batch_decode(*args, **kargs)
        return [text.split(" ") for text in batch_texts]

    @classmethod
    def preprocess(cls, raw_text: list[str]):
        text = list(map(lambda a: a.translate(cls.accents_table), raw_text))
        text = list(map(lambda a: cls.special_tokens.get(a, a), text))
        return text
