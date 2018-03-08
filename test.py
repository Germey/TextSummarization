from preprocess.config import ENABLE_PIPELINES
import json

file = './data/nlpcc/toutiao4nlpcc_eval/evaluation_with_ground_truth.txt'
f = open(file, encoding='utf-8')
max = 10
pipelines = [pipeline() for pipeline in ENABLE_PIPELINES]

for index, line in enumerate(f.readlines()):
    data = json.loads(line)
    article = data.get('article')
    for pipeline in pipelines:
        article = pipeline.process_text(article)
    print(article)
