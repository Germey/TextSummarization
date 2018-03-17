import logging
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

source = './dataset/nlpcc/articles.pretrain.txt'

sentences = LineSentence(source)

# output model
output1 = './word2vec/word2vec.model'
# output vector
output2 = './word2vec/word2vec.vector'


def train():
    # train word2vec
    print('Start training...')
    model = Word2Vec(sentences=sentences, min_count=1, size=200, workers=multiprocessing.cpu_count())
    
    # save word2vec
    print('Save model to', output1)
    model.save(output1)
    print('Save model to', output2)
    model.wv.save_word2vec_format(output2)
    