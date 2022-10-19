import json
import logging
import random
from functools import reduce
from typing import List, TextIO, Union

from flair.data import Corpus, FlairDataset, Sentence
from torch.utils.data import Dataset
from torch.utils.data.dataset import Subset
from tqdm import tqdm

import config

logger = logging.getLogger(__name__)


class FormatSentence(Sentence):
    def __init__(self, text: Union[str, List[str]], entities: list[dict] = None, *args, **kargs):
        super(FormatSentence, self).__init__(text, *args, **kargs)
        self.entities = entities
        self._nested_pairs = None

    @property
    def sentence_tokens(self):
        return list(map(lambda a: a.text, self.tokens))

    @property
    def previous_tokens(self):
        return list(map(lambda a: a.text, self.previous_sentence().tokens))

    @property
    def next_tokens(self):
        return list(map(lambda a: a.text, self.next_sentence().tokens))

    @property
    def nested_entities(self):
        entities = list()
        for types_pairs in self.nested_pairs.values():
            for pair in types_pairs:
                entities.extend(pair)
        return reduce(lambda x, y: x if y in x else x + [y], [[], ] + entities)

    @property
    def flat_entities(self):
        nested_entity, flat_entities = list(), list()
        for types_pairs in self.nested_pairs.values():
            for pair in types_pairs:
                nested_entity.extend(pair)
        for entity in self.entities:
            if entity not in nested_entity:
                flat_entities.append(entity)
        return flat_entities

    def get_nested_entities(self, kind: str):
        entities = list()
        for pair in self.nested_pairs.get(kind, []):
            entities.extend(pair)
        return reduce(lambda x, y: x if y in x else x + [y], [[], ] + entities)

    @property
    def nested_pairs(self) -> dict:
        def check_nested(entity_1, entity_2) -> Union[str, None]:
            s_1, e_1, t_1 = entity_1['start'], entity_1['end'] - 1, entity_1['type']
            s_2, e_2, t_2 = entity_2['start'], entity_2['end'] - 1, entity_2['type']
            if s_1 == s_2 and e_1 == e_2 and t_1 != t_2:
                return "ME"
            elif e_2 >= e_1 >= s_1 >= s_2 and t_1 == t_2:
                return "NST"
            elif e_2 >= e_1 >= s_1 >= s_2 and t_1 != t_2:
                return "NDT"
            elif e_1 > e_2 >= s_1 > s_2 and t_1 == t_2:
                return "OST"
            elif e_1 > e_2 >= s_1 > s_2 and t_1 != t_2:
                return "ODT"
            else:
                return None

        if self._nested_pairs is None:
            self._nested_pairs = dict()
            for i, entity_1 in enumerate(self.entities):
                for entity_2 in self.entities[i + 1:]:
                    nested_type = check_nested(entity_1, entity_2)
                    if nested_type is not None:
                        pairs = self._nested_pairs.get(nested_type, list())
                        self._nested_pairs[nested_type] = pairs + [[entity_1, entity_2]]
                    else:
                        nested_type = check_nested(entity_2, entity_1)
                        if nested_type is not None:
                            pairs = self._nested_pairs.get(nested_type, list())
                            self._nested_pairs[nested_type] = pairs + [[entity_1, entity_2]]
        return self._nested_pairs


class FormatDataset(FlairDataset):
    def __init__(self, file: TextIO, name: str, mini_size: int = None):
        self.name = name
        data_list = json.load(file)
        if mini_size is not None:
            data_list = random.sample(data_list, mini_size)
        self._sentences = self.format(data_list)

    def format(self, data_list: list[dict]) -> list[FormatSentence]:
        sentences = list()
        for data in tqdm(data_list, desc=f"Formatting {self.name}"):
            sentence = FormatSentence(data['tokens'])
            sentence._previous_sentence = FormatSentence(data['ltokens'])
            sentence._next_sentence = FormatSentence(data['rtokens'])
            sentence.add_label('pos', data['pos'])
            if 'tags' in data.keys():
                sentence.add_label('tags', data['tags'])
            sentence.entities = data['entities']
            sentences.append(sentence)

        return sentences

    def add_data(self, sentences: list[Sentence]):
        self._sentences += sentences

    def get_nested_entities_num(self, kind: str):
        return sum(map(lambda a: len(a.get_nested_entities(kind)), self._sentences))

    @property
    def sentences(self):
        return self._sentences

    @property
    def entities_num(self):
        return sum(map(lambda a: len(a.entities), self._sentences))

    @property
    def entities_num_by_length(self):
        nums = dict()
        for sentence in self.sentences:
            for entity in sentence.entities:
                entity_length = entity['end'] - entity['start']
                if entity_length > 8:
                    entity_length = 9
                nums[entity_length] = nums.get(entity_length, 0) + 1
        return nums

    @property
    def nested_entities_num(self):
        return sum(map(lambda a: len(a.nested_entities), self._sentences))

    @property
    def flat_entities_num(self):
        return sum(map(lambda a: len(a.flat_entities), self._sentences))

    @property
    def average_sentences_length(self):
        return sum(map(len, self.sentences)) / len(self.sentences)

    @property
    def average_entities_length(self):
        return sum(map(lambda a: sum(map(lambda a: a['end'] - a['start'], a.entities)), self.sentences)) / self.entities_num

    def __getitem__(self, index: int) -> FormatSentence:
        return self._sentences[index]

    def __len__(self) -> int:
        return len(self._sentences)

    def is_in_memory(self) -> bool:
        return True


class FormatCorpus(Corpus):
    def __init__(self, dataset_name: str, concat: bool = False, max_length: int = None, mini_size: int = None):
        dataset_path = f"{config.ADVANCED_OPTIONS.root_path}/data/{dataset_name}"

        try:
            super(FormatCorpus, self).__init__(name=dataset_name)
        except RuntimeError:
            pass

        self._train: FormatDataset = self.create_dataset(dataset_path, 'train', mini_size)
        self._dev: FormatDataset = self.create_dataset(dataset_path, 'dev')
        self._test: FormatDataset = self.create_dataset(dataset_path, 'test')

        with open(f"{dataset_path}/processed/metadata.json", encoding='utf-8') as file:
            metadata = json.load(file)
        types = metadata.pop('type')
        types_idx = range(len(types))
        self.metadata = {
            'types2idx': dict(zip(types, types_idx)),
            'idx2types': dict(zip(types_idx, types)),
            'chars_list': metadata['char'],
            'pos_list': metadata['pos']
        }

        if concat:
            self._train.add_data(self._dev.sentences)
            self._dev = None

        if max_length is not None:
            self.filter(lambda a: len(a) < max_length, ['train'])

    @staticmethod
    def create_dataset(dataset_path: str, mode: str, mini_size: int = None):
        try:
            with open(f"{dataset_path}/processed/{mode}.json", encoding="utf-8") as file:
                dataset = FormatDataset(file, mode, mini_size)
        except FileNotFoundError as _:
            return None
        return dataset

    def filter(self, condition: callable, dataset_names: list[str] = None):
        if dataset_names is None:
            dataset_names = ['train']

        logger.info("Filtering long sentences")
        for name in dataset_names:
            dataset = self.__dict__[f'_{name}']
            self.__dict__[f'_{name}'] = self._filter(dataset, condition)
        logger.info(self)

    @staticmethod
    def _filter(dataset: FormatDataset, condition: callable) -> Dataset:

        # find out empty sentence indices
        empty_sentence_indices = []
        non_empty_sentence_indices = []
        index = 0

        for sentence in dataset:
            if condition(sentence):
                non_empty_sentence_indices.append(index)
            else:
                empty_sentence_indices.append(index)
            index += 1

        # create subset of non-empty sentence indices
        subset = Subset(dataset, non_empty_sentence_indices)
        logger.info(f'{dataset.name}: filtered sentences num:{len(empty_sentence_indices)}')

        return subset

    def get_nested_entities_nums(self, kind: str):
        return {
            'train': self.train.get_nested_entities_num(kind),
            'dev': self.dev.get_nested_entities_num(kind),
            'test': self.test.get_nested_entities_num(kind),
        }

    @property
    def datasets(self) -> list[FormatDataset]:
        datasets = map(lambda a: self.__dict__[f'_{a}'], ['train', 'dev', 'test'])
        return list(filter(lambda a: a is not None, datasets))

    @property
    def train(self) -> FormatDataset:
        return self._train

    @property
    def dev(self) -> FormatDataset:
        return self._dev

    @property
    def test(self) -> FormatDataset:
        return self._test

    @property
    def sentences_nums(self):
        return {
            'train': len(self.train),
            'dev': len(self.dev),
            'test': len(self.test),
        }

    @property
    def entities_nums(self):
        return {
            'train': self.train.entities_num,
            'dev': self.dev.entities_num,
            'test': self.test.entities_num,
        }

    @property
    def entities_nums_by_length(self):
        return {
            'train': self.train.entities_num_by_length,
            'dev': self.dev.entities_num_by_length,
            'test': self.test.entities_num_by_length,
        }

    @property
    def nested_entities_nums(self):
        return {
            'train': self.train.nested_entities_num,
            'dev': self.dev.nested_entities_num,
            'test': self.test.nested_entities_num,
        }

    @property
    def flat_entities_nums(self):
        return {
            'train': self.train.flat_entities_num,
            'dev': self.dev.flat_entities_num,
            'test': self.test.flat_entities_num,
        }

    @property
    def average_sentence_lengths(self):
        return {
            'train': self.train.average_sentences_length,
            'dev': self.dev.average_sentences_length,
            'test': self.test.average_sentences_length,
        }

    @property
    def average_entities_lengths(self):
        return {
            'train': self.train.average_entities_length,
            'dev': self.dev.average_entities_length,
            'test': self.test.average_entities_length,
        }

    def __len__(self):
        return len(self._train)


def corpus_statistic(corpus: FormatCorpus):
    print(f"Name:{corpus.name}")
    print(f"sentences_nums:{corpus.sentences_nums}")
    print(f"entities_nums:{corpus.entities_nums}")
    print(f"entities_nums_by_length:{corpus.entities_nums_by_length}")
    print(f"nested_entities_nums:{corpus.nested_entities_nums}")
    print(f"flat_entities_nums:{corpus.flat_entities_nums}")
    print(f"ME:{corpus.get_nested_entities_nums('ME')}")
    print(f"NST:{corpus.get_nested_entities_nums('NST')}")
    print(f"NDT:{corpus.get_nested_entities_nums('NDT')}")
    print(f"OST:{corpus.get_nested_entities_nums('OST')}")
    print(f"ODT:{corpus.get_nested_entities_nums('ODT')}")
    print(f"average_sentence_lengths:{corpus.average_sentence_lengths}")
    print(f"average_entities_lengths:{corpus.average_entities_lengths}")


if __name__ == '__main__':
    corpus_statistic(FormatCorpus(**vars(config.DATA_OPTIONS)))
