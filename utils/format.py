import json
import logging

from tqdm import tqdm

import config

logger = logging.getLogger(__name__)


class Formatter:
    def __init__(self, dataset_path: str):
        self._metadata = None
        self._dataset_path = dataset_path
        self._train = None
        self._dev = None
        self._test = None

    @property
    def dataset_path(self):
        return self._dataset_path

    @property
    def train(self):
        raise NotImplemented

    @property
    def dev(self):
        raise NotImplemented

    @property
    def test(self):
        raise NotImplemented

    @property
    def metadata(self):
        if self._metadata is None:
            self._metadata = {
                'pos': set(),
                'char': set(),
                'type': set(),
            }

            for divide in ['train', 'dev', 'test']:
                with open(f"{self._dataset_path}/processed/{divide}.json", 'r', encoding='utf-8') as file:
                    for data in tqdm(json.load(file), desc=divide):
                        self._metadata['pos'].update(set(data['pos']))
                        for token in data['tokens']:
                            self._metadata['char'].update(set(token))
                        for token in data['ltokens']:
                            self._metadata['char'].update(set(token))
                        for token in data['rtokens']:
                            self._metadata['char'].update(set(token))
                        for entity in data['entities']:
                            self._metadata['type'].add(entity['type'])

            for key, value in self._metadata.items():
                self._metadata[key] = list(value)

        return self._metadata

    def dump(self):
        with open(f"{self.dataset_path}/processed/metadata.json", 'w', encoding='utf-8') as file:
            json.dump(self.metadata, file, ensure_ascii=False)
        with open(f"{self.dataset_path}/processed/train.json", 'w', encoding='utf-8') as file:
            json.dump(self.train, file, ensure_ascii=False)
        with open(f"{self.dataset_path}/processed/dev.json", 'w', encoding='utf-8') as file:
            json.dump(self.dev, file, ensure_ascii=False)
        with open(f"{self.dataset_path}/processed/test.json", 'w', encoding='utf-8') as file:
            json.dump(self.test, file, ensure_ascii=False)


class Conll2003Formatter(Formatter):
    def __init__(self, dataset_path: str):
        super().__init__(dataset_path)

    @property
    def train(self):
        if self._train is None:
            self._train = self.format("train")
        return self._train

    @property
    def dev(self):
        if self._dev is None:
            self._dev = self.format("valid")
        return self._dev

    @property
    def test(self):
        if self._test is None:
            self._test = self.format("test")
        return self._test

    def format(self, divide: str) -> list[dict]:
        data_list = list()
        with open(f"{self.dataset_path}/raw/{divide}.txt", 'r', encoding='utf-8') as file:
            doc_start = True
            tokens, pos, tags = list(), list(), list()
            for line in tqdm(file, desc=divide):
                line = line.strip('\n')
                if line == '-DOCSTART- -X- -X- O':
                    doc_start = True
                elif line == '':
                    if len(tokens) != 0:
                        data = {
                            'tokens': tokens,
                            'pos': pos,
                            'tags': tags,
                            'entities': self.tag_transform(tokens, tags)
                        }

                        if doc_start:
                            doc_start = False
                            data['ltokens'] = list()
                            if len(data_list) > 0:
                                data_list[-1]['rtokens'] = list()
                        else:
                            data_list[-1]['rtokens'] = tokens
                            data['ltokens'] = data_list[-1]['tokens']

                        data_list.append(data)
                        tokens, pos, tags = list(), list(), list()
                else:
                    contents = line.split(' ')
                    tokens.append(contents[0])
                    pos.append(contents[1])
                    tags.append(contents[-1])

            data_list[-1]['rtokens'] = list()
            return data_list

    @staticmethod
    def tag_transform(tokens: list[str], tags: list[str]) -> list[dict]:
        """
        extract entities from the labeled text
        :param tokens: raw text
        :param tags: (sentence_length,) labels
        :return: entities indicated by labels
        """
        entities, entity_type = [], None
        for index, tag in enumerate(tags):
            if tag[0] == 'B':
                start, end = index, index + 1
                entity_type = (tag[2:]).lower()
                while end + 1 < len(tokens) and tags[end] == 'I' + tag[1:]:
                    end += 1
                entities.append({
                    "start": start,
                    "end": end,
                    "type": entity_type,
                })
        return entities


class WeiboNERFormatter(Formatter):
    def __init__(self, dataset_path: str):
        super().__init__(dataset_path)

    @property
    def train(self):
        if self._train is None:
            self._train = self.format("train")
        return self._train

    @property
    def dev(self):
        if self._dev is None:
            self._dev = self.format("dev")
        return self._dev

    @property
    def test(self):
        if self._test is None:
            self._test = self.format("test")
        return self._test

    def format(self, divide: str) -> list[dict]:
        data_list = list()
        with open(f"{self.dataset_path}/raw/{divide}_word_bmes_pos.txt", 'r', encoding='utf-8') as file:
            tokens, pos, tags = list(), list(), list()
            for line in tqdm(file, desc=divide):
                line = line.strip('\n')
                if line == '':
                    if len(tokens) != 0:
                        data_list.append({
                            'tokens': tokens,
                            'pos': pos,
                            'tags': tags,
                            'entities': self.tag_transform(tokens, tags),
                            'ltokens': [],
                            'rtokens': [],
                        })
                        tokens, pos, tags = list(), list(), list()
                else:
                    contents = line.split(' ')
                    tokens.append(contents[0])
                    pos.append(contents[1])
                    tags.append(contents[-1])

            data_list[-1]['rtokens'] = list()
            return data_list

    @staticmethod
    def tag_transform(tokens: list[str], tags: list[str]) -> list[dict]:
        """
        extract entities from the labeled text
        :param tokens: raw text
        :param tags: (sentence_length,) labels
        :return: entities indicated by labels
        """
        entities, entity_type = [], None
        for index, tag in enumerate(tags):
            start, end = index, index + 1
            entity_type = (tag[2:]).lower()
            if tag[0] == 'B':
                while end + 1 < len(tokens) and tags[end + 1] == 'M' + tag[1:]:
                    end += 1
                entities.append({
                    "start": start,
                    "end": end + 1,
                    "type": entity_type,
                })
            elif tag[0] == 'S':
                entities.append({
                    "start": start,
                    "end": end,
                    "type": entity_type,
                })
        return entities


if __name__ == '__main__':
    formatter = Conll2003Formatter(f"{config.ADVANCED_OPTIONS.root_path}/data/conll2003")
    formatter.dump()
