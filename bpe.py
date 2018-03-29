import re


def process_raw_words(words, endtag='-'):
    '''把单词分割成最小的符号，并且加上结尾符号'''
    vocabs = {}
    for word, count in words.items():
        # 加上空格
        word = re.sub(r'([a-zA-Z])', r' \1', word)
        word += ' ' + endtag
        vocabs[word] = count
    return vocabs


def get_symbol_pairs(vocabs):
    ''' 获得词汇中所有的字符pair，连续长度为2，并统计出现次数
    Args:
        vocabs: 单词dict，(word, count)单词的出现次数。单词已经分割为最小的字符
    Returns:
        pairs: ((符号1, 符号2), count)
    '''
    # pairs = collections.defaultdict(int)
    pairs = dict()
    for word, freq in vocabs.items():
        # 单词里的符号
        symbols = word.split()
        for i in range(len(symbols) - 1):
            p = (symbols[i], symbols[i + 1])
            pairs[p] = pairs.get(p, 0) + freq
    return pairs


def merge_symbols(symbol_pair, vocabs):
    '''把vocabs中的所有单词中的'a b'字符串用'ab'替换
    Args:
        symbol_pair: (a, b) 两个符号
        vocabs: 用subword(symbol)表示的单词，(word, count)。其中word使用subword空格分割
    Returns:
        vocabs_new: 替换'a b'为'ab'的新词汇表
    '''
    vocabs_new = {}
    raw = ' '.join(symbol_pair)
    merged = ''.join(symbol_pair)
    # 非字母和数字字符做转义
    bigram = re.escape(raw)
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word, count in vocabs.items():
        word_new = p.sub(merged, word)
        vocabs_new[word_new] = count
    return vocabs_new


raw_words = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
vocabs = process_raw_words(raw_words)

num_merges = 10
print(vocabs)
for i in range(num_merges):
    pairs = get_symbol_pairs(vocabs)
    # 选择出现频率最高的pair
    symbol_pair = max(pairs, key=pairs.get)
    vocabs = merge_symbols(symbol_pair, vocabs)
print(vocabs)
