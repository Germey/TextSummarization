from config import ENABLE_PIPELINES, VOCABS_SIZE_LIMIT, DATASET_OUTPUT_FOLDER
from preprocess.writer import Writer
from preprocess.vocab import VocabTransformer
import json
from os.path import exists, join
from os import makedirs

if not exists(DATASET_OUTPUT_FOLDER):
    makedirs(DATASET_OUTPUT_FOLDER)

# file = './data/nlpcc/toutiao4nlpcc/train_without_summ.txt'

# pipelines and writer to process data
pipelines = [pipeline() for pipeline in ENABLE_PIPELINES]
writer = Writer(folder=DATASET_OUTPUT_FOLDER)
vocab_transformer = VocabTransformer(limit=VOCABS_SIZE_LIMIT)

# train
file = './source/nlpcc/toutiao4nlpcc/train_with_summ.txt'
f = open(file, encoding='utf-8')

# start processing
articles = []
summaries = []

# get source data
for line in f.readlines():
    item = json.loads(line)
    article = item.get('article')
    summary = item.get('summarization')
    articles.append(article)
    summaries.append(summary)

# pre precess by pipeline
for pipeline in pipelines:
    print('Running', pipeline)
    articles = pipeline.process_all(articles)
    summaries = pipeline.process_all(summaries)

# get vocabs of articles and summaries, they use the same vocabs
word2id, id2word = vocab_transformer.build_vocabs(articles)

# write data to txt
writer.write_to_txt(articles, 'articles.train.txt')
writer.write_to_txt(summaries, 'summaries.train.txt')

# write vocab to json
writer.write_to_json(word2id, 'articles_vocabs.json')
writer.write_to_json(word2id, 'summaries_vocabs.json')

# eval
file = './source/nlpcc/toutiao4nlpcc_eval/evaluation_with_ground_truth.txt'
f = open(file, encoding='utf-8')

# start processing
articles = []
summaries = []

# get source data
for line in f.readlines():
    item = json.loads(line)
    article = item.get('article')
    summary = item.get('summarization')
    articles.append(article)
    summaries.append(summary)

# pre precess by pipeline
for pipeline in pipelines:
    print('Running', pipeline)
    articles = pipeline.process_all(articles)
    summaries = pipeline.process_all(summaries)

# write data to txt
writer.write_to_txt(articles, 'articles.eval.txt')
writer.write_to_txt(summaries, 'summaries.eval.txt')
