import sys

sys.path.append('./data')

from data_iterator import BiTextIterator, TextIterator
from train import FLAGS


def main():
    train_set = BiTextIterator(source=FLAGS.source_train_data,
                               target=FLAGS.target_train_data,
                               source_dict=FLAGS.source_vocabulary,
                               target_dict=FLAGS.target_vocabulary,
                               batch_size=FLAGS.batch_size,
                               maxlen=FLAGS.max_seq_length,
                               n_words_source=FLAGS.num_encoder_symbols,
                               n_words_target=FLAGS.num_decoder_symbols,
                               shuffle_each_epoch=FLAGS.shuffle_each_epoch,
                               sort_by_length=FLAGS.sort_by_length,
                               maxibatch_size=FLAGS.max_load_batches)
    
    for source, target in train_set.next():
        print(source, target)


if __name__ == '__main__':
    main()
