from config import ENABLE_PIPELINES, VOCABS_SIZE_LIMIT, DATASET_OUTPUT_FOLDER
from preprocess.writer import Writer
from preprocess.vocab import VocabTransformer
import json

# file = './data/nlpcc/toutiao4nlpcc/train_without_summ.txt'
file = './source/nlpcc/toutiao4nlpcc_eval/evaluation_with_ground_truth.txt'
f = open(file, encoding='utf-8')

# pipelines and writer to process data
pipelines = [pipeline() for pipeline in ENABLE_PIPELINES]
writer = Writer(folder=DATASET_OUTPUT_FOLDER)
vocab_transformer = VocabTransformer(limit=VOCABS_SIZE_LIMIT)

# start processing
articles = []
summaries = []

# get source data
for index, line in enumerate(f.readlines()):
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
writer.write_to_txt(articles, 'articles.txt')
writer.write_to_txt(summaries, 'summaries.txt')

# write vocab to json
writer.write_to_json(word2id, 'articles_vocabs.json')
writer.write_to_json(word2id, 'summaries_vocabs.json')
