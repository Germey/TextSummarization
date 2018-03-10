from preprocess.iterator import BiTextIterator, TextIterator
from train import FLAGS


def main():
    train_set = TextIterator(source=FLAGS.source_train_data,
                             source_dict=FLAGS.source_vocabulary,
                             batch_size=FLAGS.batch_size,
                             max_length=FLAGS.max_seq_length,
                             n_words_source=FLAGS.num_encoder_symbols,
                             sort_by_length=FLAGS.sort_by_length,
                             )
    for source in train_set.next():
        print('S', len(source))


if __name__ == '__main__':
    main()
