import argparse
import os
import sys
from os import walk
from os.path import join, exists
from os import makedirs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="learn BPE-based word segmentation")
    
    parser.add_argument(
        '--input', '-i', type=str,
        help="Input text (default: standard input).")
    
    parser.add_argument(
        '--output', '-o', type=str,
        help="Output file for BPE codes (default: standard output)")
    parser.add_argument(
        '--symbols', '-s', type=int, default=10000,
        help="Create this many new symbols (each representing a character n-gram) (default: %(default)s))")
    parser.add_argument(
        '--min-frequency', type=int, default=2, metavar='FREQ',
        help='Stop if no symbol pair has frequency >= FREQ (default: %(default)s))')
    parser.add_argument('--dict-input', action="store_true",
                        help="If set, input file is interpreted as a dictionary where each line contains a word-count pair")
    parser.add_argument(
        '--verbose', '-v', action="store_true",
        help="verbose mode.")
    
    args = parser.parse_args()
    
    input = args.input
    
    output = args.output
    
    if not exists(output):
        makedirs(output)
    
    learn_bpe_cmd = 'python3 learn_bpe.py -s %(symbols)s -i %(input)s -o %(output)s' % (
        {'symbols': args.symbols, 'input': join(input, 'articles.train.txt'), 'output': join(output, 'codes.txt')})
    print(learn_bpe_cmd)
    os.system(learn_bpe_cmd)
    
    # apply bpe
    files = [
        'articles.train.txt',
        'articles.eval.txt',
        'summaries.train.txt',
        'summaries.eval.txt'
    ]
    for file in files:
        apply_bpe_cmd = 'python3 apply_bpe.py -c %(codes)s -i %(input)s -o %(output)s' % (
            {
                'codes': join(output, 'codes.txt'),
                'input': join(input, file),
                'output': join(output, file)
            }
        )
        print(apply_bpe_cmd)
        os.system(apply_bpe_cmd)
    
    # vocab
    vocab_bpe_cmd = 'python3 get_vocab.py < %(train)s  > %(vocab)s' % (
        {
            'train': join(output, 'articles.train.txt'),
            'vocab': join(output, 'vocab.txt')
        }
    )
    print(vocab_bpe_cmd)
    os.system(vocab_bpe_cmd)
    
    import json
    
    d = {}
    
    d['GO'] = 0
    d['EOS'] = 1
    d['UNK'] = 2
    
    count = 3
    
    with open(join(output, 'vocab.txt'), encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            word = line.split()[0]
            d[word] = count
            count += 1
    
    with open(join(output, 'vocab.json'), 'w', encoding='utf-8') as f:
        f.write(json.dumps(d, ensure_ascii=False))
