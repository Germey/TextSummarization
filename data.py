from preprocess.iterator import BiTextIterator, TextIterator
from train import FLAGS


def main():
    train_set = TextIterator(source=FLAGS.source_valid_data,
                             source_dict=FLAGS.source_vocabulary,
                             batch_size=FLAGS.batch_size,
                             max_length=FLAGS.max_seq_length,
                             n_words_source=FLAGS.num_encoder_symbols,
                             sort_by_length=FLAGS.sort_by_length,
                             )
    for source in train_set.next():
        print('S', len(source))
    print(train_set.length())
    
    train_set = BiTextIterator(source=FLAGS.source_valid_data,
                               target=FLAGS.target_valid_data,
                               source_dict=FLAGS.source_vocabulary,
                               target_dict=FLAGS.target_vocabulary,
                               batch_size=FLAGS.batch_size,
                               max_length=FLAGS.max_seq_length,
                               sort_by_length=FLAGS.sort_by_length,
                               )
    for source, target in train_set.next():
        print('S', len(source), len(target))
    print(train_set.length())


if __name__ == '__main__':
    main()
