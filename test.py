from preprocess.config import ENABLE_PIPELINES
import json

file = './data/nlpcc/toutiao4nlpcc_eval/evaluation_with_ground_truth.txt'
f = open(file, encoding='utf-8')
max = 10
pipelines = [pipeline() for pipeline in ENABLE_PIPELINES]

data = []

for index, line in enumerate(f.readlines()):
    item = json.loads(line)
    article = item.get('article')
    data.append(article)

for pipeline in pipelines:
    print(pipeline.__str__())
    data = pipeline.process_all(data)

print(data)
