import time
from tqdm import tqdm

from preprocess.iterator import BiTextIterator, TextIterator
from train import FLAGS

batch_size = 128


def main():
    train_set = BiTextIterator(source=FLAGS.source_valid_data,
                               target=FLAGS.target_valid_data,
                               source_dict=FLAGS.source_vocabulary,
                               target_dict=FLAGS.target_vocabulary,
                               batch_size=batch_size,
                               max_length=FLAGS.max_seq_length,
                               n_words_source=FLAGS.num_encoder_symbols,
                               n_words_target=FLAGS.num_decoder_symbols,
                               sort_by_length=FLAGS.sort_by_length,
                               split_sign=FLAGS.split_sign,
                               )
    with tqdm(total=train_set.length()) as pbar:
        train_set.reset()
        processed_length = batch_size
        for source, target in train_set.next():
            processed_length += len(source)
            print('Length', len(source), len(target), processed_length)
            time.sleep(1)
            pbar.update(len(source))
            
            print('Source 0', list(source[0]), len(source[0]))
            print('Source 0', list(target[0]), len(target[0]))
    
    train_set.reset()
    
    print('Reset')
    
    for source, target in train_set.next():
        print('Length', len(source), len(target))


if __name__ == '__main__':
    main()
