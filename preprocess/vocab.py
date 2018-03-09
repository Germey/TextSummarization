from itertools import chain
import pandas as pd

from preprocess import config


class VocabFactory():
    def split_lines(self, data):
        """
        split lines to array
        :param data: 你好\t啊
        :return: ['你好', '啊']
        """
        result = []
        for line in data:
            result.append(line.split(config.SEGMENT_JOIN_FLAG))
        return result
    
    def build_vocab(self, data):
        """
        build vocab
        :param data:
        :return:
        """
        # merge all words
        all_words = list(chain(*data))
        # all words to Series
        all_words_sr = pd.Series(all_words)
        # get value count, index changed to set
        all_words_counts = all_words_sr.value_counts()
        # Get words set
        all_words_set = all_words_counts.index
        # Get words ids
        all_words_ids = len(all_words_set)
        
        # Dict to transform
        word2id = pd.Series(all_words_ids, index=all_words_set)
        id2word = pd.Series(all_words_set, index=all_words_ids)
        
        return word2id, id2word
    
    def get_vocab(self, data):
        """
        get vocab
        :param data:
        :return:
        """
        data = self.split_lines(data)
        word2id, id2word = self.build_vocab(data)
        return word2id, id2word
