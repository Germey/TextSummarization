file = '/private/var/py/TextSummarization/source/nlpcc/toutiao4nlpcc/train_with_summ.txt'

import numpy as np
import json

with open(file) as f:
    lines = []
    for line in f.readlines():
        lines.append(len(json.loads(line).get('article')))
    
    print(lines)
    
    print(np.mean(lines))
    print(max(lines))
    print(min(lines))