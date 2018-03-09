from preprocess.config import ENABLE_PIPELINES
from preprocess.writer import Writer
import json

# file = './data/nlpcc/toutiao4nlpcc/train_without_summ.txt'
file = './source/nlpcc/toutiao4nlpcc_eval/evaluation_with_ground_truth.txt'
f = open(file, encoding='utf-8')

# pipelines and writer to process data
pipelines = [pipeline() for pipeline in ENABLE_PIPELINES]
writer = Writer()

# start processing
articles = []
summaries = []

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

# write to txt
writer.write_to_txt(articles, 'articles.txt')
writer.write_to_txt(summaries, 'summaries.txt')
