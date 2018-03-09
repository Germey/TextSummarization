import numpy as np
import json
import pickle as pkl
from preprocess import config

extra_tokens = [config.GO, config.EOS, config.UNK]

start_token = extra_tokens.index(config.GO)  # start_token = 0
end_token = extra_tokens.index(config.EOS)  # end_token = 1
unk_token = extra_tokens.index(config.UNK)


def load_dict(filename):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except:
        with open(filename, 'r') as f:
            return pkl.load(f)


class TextIterator():
    """Simple Text iterator."""
    
    def __init__(self, source, source_dict,
                 batch_size=128, max_length=None,
                 n_words_source=-1,
                 skip_empty=False,
                 sort_by_length=False,
                 encoding='utf-8'
                 ):
        
        self.source = open(source, 'r', encoding=encoding)
        self.source_dict = load_dict(source_dict)
        self.batch_size = batch_size
        self.max_length = max_length
        self.skip_empty = skip_empty
        self.n_words_source = n_words_source
        
        if self.n_words_source > 0:
            for key, idx in self.source_dict.items():
                if idx >= self.n_words_source:
                    del self.source_dict[key]
        
        self.sort_by_length = sort_by_length
        self.source_buffer = []
        self.end_of_data = False
    
    def __len__(self):
        return sum([1 for _ in self.next()])
    
    def reset(self):
        
        self.source.seek(0)
    
    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration
        
        source = []
        
        # fill buffer, if it's empty
        if len(self.source_buffer) == 0:
            for k_ in range(self.k):
                ss = self.source.readline()
                if ss == "":
                    break
                self.source_buffer.append(ss.strip().split())
            
            # sort by buffer
            if self.sort_by_length:
                slen = np.array([len(s) for s in self.source_buffer])
                sidx = slen.argsort()
                
                sbuf = [self.source_buffer[i] for i in sidx]
                
                self.source_buffer = sbuf
            else:
                self.source_buffer.reverse()
        
        if len(self.source_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration
        
        try:
            # actual work here
            while True:
                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break
                ss = [self.source_dict[w] if w in self.source_dict
                      else unk_token for w in ss]
                
                if self.max_length and len(ss) > self.max_length:
                    continue
                if self.skip_empty and (not ss):
                    continue
                source.append(ss)
                
                if len(source) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True
        
        if len(source) == 0:
            source = self.next()
        
        yield source


class BiTextIterator():
    """Simple Bi text iterator."""
    
    def __init__(self, source, target,
                 source_dict, target_dict,
                 batch_size=128,
                 max_length=100,
                 n_words_source=-1,
                 n_words_target=-1,
                 skip_empty=False,
                 shuffle_each_epoch=False,
                 sort_by_length=True,
                 encoding='utf-8'):
        
        self.source = open(source, 'r')
        self.target = open(target, 'r')
        
        self.source_dict = load_dict(source_dict)
        self.target_dict = load_dict(target_dict)
        
        self.batch_size = batch_size
        self.max_length = max_length
        self.skip_empty = skip_empty
        
        self.n_words_source = n_words_source
        self.n_words_target = n_words_target
        
        print(n_words_source, n_words_target)
        
        if self.n_words_source > 0:
            for key, idx in self.source_dict.items():
                if idx >= self.n_words_source:
                    del self.source_dict[key]
        
        if self.n_words_target > 0:
            for key, idx in self.target_dict.items():
                if idx >= self.n_words_target:
                    del self.target_dict[key]
        
        self.shuffle = shuffle_each_epoch
        self.sort_by_length = sort_by_length
        
        self.source_buffer = []
        self.target_buffer = []
        
        self.end_of_data = False
    
    def __len__(self):
        return sum([1 for _ in self.next()])
    
    def reset(self):
        self.source.seek(0)
        self.target.seek(0)
    
    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration
        
        source = []
        target = []
        
        assert len(self.source_buffer) == len(self.target_buffer), 'Buffer size mismatch!'
        
        if len(self.source_buffer) == 0:
            for k_ in range(self.k):
                ss = self.source.readline()
                if ss == "":
                    break
                tt = self.target.readline()
                if tt == "":
                    break
                self.source_buffer.append(ss.strip().split())
                self.target_buffer.append(tt.strip().split())
            
            # sort by target buffer
            if self.sort_by_length:
                tlen = np.array([len(t) for t in self.target_buffer])
                tidx = tlen.argsort()
                
                sbuf = [self.source_buffer[i] for i in tidx]
                tbuf = [self.target_buffer[i] for i in tidx]
                
                self.source_buffer = sbuf
                self.target_buffer = tbuf
            
            else:
                self.source_buffer.reverse()
                self.target_buffer.reverse()
        
        if len(self.source_buffer) == 0 or len(self.target_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration
        
        try:
            
            # actual work here
            while True:
                
                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break
                ss = [self.source_dict[w] if w in self.source_dict
                      else unk_token for w in ss]
                
                # read from source file and map to word index
                tt = self.target_buffer.pop()
                tt = [self.target_dict[w] if w in self.target_dict
                      else unk_token for w in tt]
                if self.n_words_target > 0:
                    tt = [w if w < self.n_words_target
                          else unk_token for w in tt]
                
                if self.max_length:
                    if len(ss) > self.max_length and len(tt) > self.max_length:
                        continue
                if self.skip_empty and (not ss or not tt):
                    continue
                
                source.append(ss)
                target.append(tt)
                
                if len(source) >= self.batch_size or \
                        len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True
        
        if len(source) == 0 or len(target) == 0:
            source, target = self.next()
        
        yield (source, target)
