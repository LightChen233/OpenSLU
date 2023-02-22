import json
import os
from collections import Counter
from collections import OrderedDict
from typing import List

import torch
from ordered_set import OrderedSet
from transformers import AutoTokenizer

from common.utils import download, unzip_file


def get_tokenizer(tokenizer_name:str):
    """auto get tokenizer

    Args:
        tokenizer_name (str): support "word_tokenizer" and other pretrained tokenizer in hugging face.

    Returns:
        Any: Tokenizer Object
    """
    if tokenizer_name == "word_tokenizer":
        return WordTokenizer(tokenizer_name)
    else:
        return AutoTokenizer.from_pretrained(tokenizer_name)

def get_tokenizer_class(tokenizer_name:str):
    """auto get tokenizer class

    Args:
        tokenizer_name (str): support "word_tokenizer" and other pretrained tokenizer in hugging face.

    Returns:
        Any: Tokenizer Class
    """
    if tokenizer_name == "word_tokenizer":
        return WordTokenizer
    else:
        return AutoTokenizer.from_pretrained

BATCH_STATE = 1
INSTANCE_STATE = 2


class WordTokenizer(object):

    def __init__(self, name):
        self.__name = name
        self.index2instance = OrderedSet()
        self.instance2index = OrderedDict()
        # Counter Object record the frequency
        # of element occurs in raw text.
        self.counter = Counter()

        self.__sign_pad = "[PAD]"
        self.add_instance(self.__sign_pad)
        self.__sign_unk = "[UNK]"
        self.add_instance(self.__sign_unk)

    @property
    def padding_side(self):
        return "right"
    @property
    def all_special_ids(self):
        return [self.unk_token_id, self.pad_token_id]

    @property
    def name_or_path(self):
        return self.__name

    @property
    def vocab_size(self):
        return len(self.instance2index)

    @property
    def pad_token_id(self):
        return self.instance2index[self.__sign_pad]

    @property
    def unk_token_id(self):
        return self.instance2index[self.__sign_unk]

    def add_instance(self, instance):
        """ Add instances to alphabet.

        1, We support any iterative data structure which
        contains elements of str type.

        2, We will count added instances that will influence
        the serialization of unknown instance.

        Args:
            instance: is given instance or a list of it.
        """

        if isinstance(instance, (list, tuple)):
            for element in instance:
                self.add_instance(element)
            return

        # We only support elements of str type.
        assert isinstance(instance, str)

        # count the frequency of instances.
        # self.counter[instance] += 1

        if instance not in self.index2instance:
            self.instance2index[instance] = len(self.index2instance)
            self.index2instance.append(instance)

    def __call__(self, instance,
                 return_tensors="pt",
                 is_split_into_words=True,
                 padding=True,
                 add_special_tokens=False,
                 truncation=True,
                 max_length=512,
                 **config):
        if isinstance(instance, (list, tuple)) and isinstance(instance[0], (str)) and is_split_into_words:
            res = self.get_index(instance)
            state = INSTANCE_STATE
        elif isinstance(instance, str) and not is_split_into_words:
            res = self.get_index(instance.split(" "))
            state = INSTANCE_STATE
        elif not is_split_into_words and isinstance(instance, (list, tuple)):
            res = [self.get_index(ins.split(" ")) for ins in instance]
            state = BATCH_STATE
        else:
            res = [self.get_index(ins) for ins in instance]
            state = BATCH_STATE
        res = [r[:max_length] if len(r) >= max_length else r for r in res]
        pad_id = self.get_index(self.__sign_pad)
        if padding and state == BATCH_STATE:
            max_len = max([len(x) for x in instance])

            for i in range(len(res)):
                res[i] = res[i] + [pad_id] * (max_len - len(res[i]))
        if return_tensors == "pt":
            input_ids = torch.Tensor(res).long()
            attention_mask = (input_ids != pad_id).long()
        elif state == BATCH_STATE:
            input_ids = res
            attention_mask = [1 if r != pad_id else 0 for batch in res for r in batch]
        else:
            input_ids = res
            attention_mask = [1 if r != pad_id else 0 for r in res]
        return TokenizedData(input_ids, token_type_ids=attention_mask, attention_mask=attention_mask)

    def get_index(self, instance):
        """ Serialize given instance and return.

        For unknown words, the return index of alphabet
        depends on variable self.__use_unk:

            1, If True, then return the index of "<UNK>";
            2, If False, then return the index of the
            element that hold max frequency in training data.

        Args:
            instance (Any): is given instance or a list of it.
        Return:
            Any: the serialization of query instance.
        """

        if isinstance(instance, (list, tuple)):
            return [self.get_index(elem) for elem in instance]

        assert isinstance(instance, str)

        try:
            return self.instance2index[instance]
        except KeyError:
            return self.instance2index[self.__sign_unk]

    def decode(self, index):
        """ Get corresponding instance of query index.

        if index is invalid, then throws exception.

        Args:
            index (int): is query index, possibly iterable.
        Returns:
            is corresponding instance.
        """

        if isinstance(index, list):
            return [self.decode(elem) for elem in index]
        if isinstance(index, torch.Tensor):
            index = index.tolist()
            return self.decode(index)
        return self.index2instance[index]
    
    def decode_batch(self, index, **kargs):
        """ Get corresponding instance of query index.

        if index is invalid, then throws exception.

        Args:
            index (int): is query index, possibly iterable.
        Returns:
            is corresponding instance.
        """
        return self.decode(index)

    def save(self, path):
        """ Save the content of alphabet to files.

        There are two kinds of saved files:
            1, The first is a list file, elements are
            sorted by the frequency of occurrence.

            2, The second is a dictionary file, elements
            are sorted by it serialized index.

        Args:
            path (str): is the path to save object.
        """
        
        with open(path, 'w', encoding="utf8") as fw:
            fw.write(json.dumps({"name": self.__name, "token_map": self.instance2index}))
    
    @staticmethod
    def from_file(path):
        with open(path, 'r', encoding="utf8") as fw:
            obj = json.load(fw)
            tokenizer = WordTokenizer(obj["name"])
            tokenizer.instance2index = OrderedDict(obj["token_map"])
            # tokenizer.counter = len(tokenizer.instance2index)
            tokenizer.index2instance = OrderedSet(tokenizer.instance2index.keys())
            return tokenizer
    
    def __len__(self):
        return len(self.index2instance)

    def __str__(self):
        return 'Alphabet {} contains about {} words: \n\t{}'.format(self.name_or_path, len(self), self.index2instance)

    def convert_tokens_to_ids(self, tokens):
        """convert token sequence to intput ids sequence

        Args:
            tokens (Any): token sequence

        Returns:
            Any: intput ids sequence
        """
        try:
            if isinstance(tokens, (list, tuple)):
                return [self.instance2index[x] for x in tokens]
            return self.instance2index[tokens]

        except KeyError:
            return self.instance2index[self.__sign_unk]


class TokenizedData():
    """tokenized output data with input_ids, token_type_ids, attention_mask
    """
    def __init__(self, input_ids, token_type_ids, attention_mask):
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask

    def word_ids(self, index: int) -> List[int or None]:
        """ get word id list

        Args:
            index (int): word index in sequence
        
        Returns:
            List[int or None]: word id list
        """
        return [j if self.attention_mask[index][j] != 0 else None for j, x in enumerate(self.input_ids[index])]

    def word_to_tokens(self, index, word_id, **kwargs):
        """map word and tokens

        Args:
            index (int): unused
            word_id (int): word index in sequence
        """
        return (word_id, word_id + 1)

    def to(self, device):
        """set device

        Args:
            device (str): support ["cpu", "cuda"]
        """
        self.input_ids = self.input_ids.to(device)
        self.token_type_ids = self.token_type_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        return self


def load_embedding(tokenizer: WordTokenizer, glove_name:str):
    """ load embedding from standford server or local cache.

    Args:
        tokenizer (WordTokenizer): non-pretrained tokenizer
        glove_name (str): _description_

    Returns:
        Any: word embedding
    """
    save_path = "save/" + glove_name + ".zip"
    if not os.path.exists(save_path):
        download("http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip#" + glove_name, save_path)
        unzip_file(save_path, "save/" + glove_name)
    dim = int(glove_name.split(".")[-2][:-1])
    embedding_list = torch.rand((tokenizer.vocab_size, dim))
    embedding_list[tokenizer.pad_token_id] = torch.zeros((1, dim))
    with open("save/" + glove_name + "/" + glove_name, "r", encoding="utf8") as f:
        for line in f.readlines():
            datas = line.split(" ")
            word = datas[0]
            embedding = torch.Tensor([float(datas[i + 1]) for i in range(len(datas) - 1)])
            tokenized = tokenizer.convert_tokens_to_ids(word)
            if isinstance(tokenized, int) and tokenized != tokenizer.unk_token_id:
                embedding_list[tokenized] = embedding

    return embedding_list
