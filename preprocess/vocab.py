from itertools import chain
import pandas as pd
import config


class VocabTransformer(object):
    def __init__(self, limit=-1):
        """
        max size of vocabs
        :param limit:
        """
        self.limit = limit
    
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
    
    def build_vocabs(self, data):
        """
        build vocabs
        :param data:
        :return:
        """
        # split lines
        data = self.split_lines(data)
        # merge all words
        all_words = list(chain(*data))
        # all words to Series
        all_words_sr = pd.Series(all_words)
        # get value count, index changed to set
        all_words_counts = all_words_sr.value_counts()
        # get words set
        all_words_set = list(all_words_counts.index)
        
        for token in (config.UNK, config.EOS, config.GO):
            all_words_set.insert(0, token)
        
        if self.limit >= 0:
            all_words_set = all_words_set[:self.limit]
        
        # get words ids
        all_words_ids = range(len(all_words_set))
        
        # dict to transform
        word2id = pd.Series(all_words_ids, index=all_words_set).to_dict()
        id2word = pd.Series(all_words_set, index=all_words_ids).to_dict()
        
        return word2id, id2word
