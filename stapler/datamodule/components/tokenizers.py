import re
from abc import ABC, abstractmethod
from typing import List, Tuple, Union


class Tokenizer(ABC):
    def __init__(self, add_special_tokens=True) -> None:
        self.vocab_dict = {}
        if add_special_tokens:
            self.special_tokens = ["[UNK]", "[SEP]", "[CLS]", "[MASK]"]
            self.pad_token = "[PAD]"

        else:
            self.special_tokens = []
            raise RuntimeError("Not supported yet")

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        pass

    @abstractmethod
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        pass

    @abstractmethod
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        pass

    @abstractmethod
    def encode(self, text: str) -> Tuple[List[int], List[int]]:
        pass


class BasicTokenizer(Tokenizer):
    def __init__(self, vocabulary: Union[str, List[str]], add_special_tokens=True) -> None:
        super().__init__(add_special_tokens)
        # Add the vocabulary to the vocab_dict
        if isinstance(vocabulary, list):
            # join the strings for each entry in the list
            vocabulary = "".join(vocabulary)
        self.vocabulary = vocabulary

        # first token is pad token
        self.vocab_dict = {self.pad_token: 0}
        self.pad_token_id = self.vocab_dict["[PAD]"]
        self.vocab_dict.update({tok: i for i, tok in enumerate(self.vocabulary, len(self.vocab_dict))})

        if add_special_tokens:
            self.vocab_dict.update({tok: i for i, tok in enumerate(self.special_tokens, len(self.vocab_dict))})
            self.unk_token_id = self.vocab_dict["[UNK]"]
            self.sep_token_id = self.vocab_dict["[SEP]"]
            self.cls_token_id = self.vocab_dict["[CLS]"]
            self.mask_token_id = self.vocab_dict["[MASK]"]

    def tokenize(self, text: str) -> List[str]:
        substrings = text.split()
        split_tokens = []
        for substring in substrings:
            if substring in self.vocab_dict:
                split_tokens.append(substring)
                continue
            for charac in substring:
                if charac.lower() in self.vocab_dict:
                    split_tokens.append(charac.lower())
                else:
                    split_tokens.extend(["[UNK]"])
        return split_tokens

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.vocab_dict[token] for token in tokens]

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.vocabulary[i] for i in ids]

    def encode(self, text: str) -> List[int]:
        "TODO: add transformers encode arguments and functionality"
        tokens = self.tokenize(text)
        token_ids = self.convert_tokens_to_ids(tokens)
        return token_ids
