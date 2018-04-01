from preprocess.writer import Writer
from preprocess.vocab import VocabTransformer
import json
from os.path import exists, join
from os import makedirs
from preprocess.pipeline import *

output_dir = join('dataset', 'nlpcc_without_date')
vocab_size_limit = 30000

# pipelines enabled
if not exists(output_dir):
    makedirs(output_dir)

# pipelines and writer to process data
pipelines = [
    StripPipeline(),
    PhonePipeline(),
    EmailPipeline(),
    UrlPipeline(),
    DatePipeline(),
    TimePipeline(),
    RemovePipeline(),
    HalfWidthPipeline(),
    LowerPipeline(),
    ReplacePipeline(),
    JiebaPipeline(),
]

writer = Writer(folder=output_dir)
vocab_transformer = VocabTransformer(limit=vocab_size_limit)

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
