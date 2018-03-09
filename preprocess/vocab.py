from itertools import chain
import pandas as pd
from preprocess import config


class VocabTransformer(object):
    
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
        all_words_set = list(all_words_counts.index)
        
        for token in (config.UNK, config.EOS, config.GO):
            all_words_set.insert(0, token)
        
        # Get words ids
        all_words_ids = range(len(all_words_set))
        
        # Dict to transform
        word2id = pd.Series(all_words_ids, index=all_words_set).to_dict()
        id2word = pd.Series(all_words_set, index=all_words_ids).to_dict()
        
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

