# !/usr/bin/env python
# coding: utf-8
from preprocess.iterator import TextIterator
from utils import prepare_batch, load_inverse_dict, seq2words
import json
import tensorflow as tf
from model import Seq2SeqModel

# Decoding parameters
tf.app.flags.DEFINE_integer('beam_width', 1, 'Beam width used in beam search')
tf.app.flags.DEFINE_integer('decode_batch_size', 1, 'Batch size used for decoding')
tf.app.flags.DEFINE_integer('max_decode_step', 50, 'Maximum time step limit to decode')
tf.app.flags.DEFINE_boolean('write_n_best', False, 'Write n-best list (n=beam_width)')
tf.app.flags.DEFINE_string('model_path', 'model/summary.ckpt-1000', 'Path to a specific model checkpoint.')
tf.app.flags.DEFINE_string('decode_input', 'dataset/nlpcc_char/articles.eval.txt', 'Decoding input path')
tf.app.flags.DEFINE_string('decode_output', 'dataset/nlpcc_char/articles.decode.txt', 'Decoding output path')

# Runtime parameters
tf.app.flags.DEFINE_boolean('allow_soft_placement', True, 'Allow device soft placement')
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Log placement of ops on devices')

FLAGS = tf.app.flags.FLAGS


def load_config(FLAGS):
    config = json.load(open('%s.json' % FLAGS.model_path, 'r'))
    for key, value in FLAGS.flag_values_dict().items():
        config[key] = value
    return config


def load_model(session, config):
    model = Seq2SeqModel(config, 'decode')
    if tf.train.checkpoint_exists(FLAGS.model_path):
        print('Reloading model parameters..')
        model.restore(session, FLAGS.model_path)
    else:
        raise ValueError(
            'No such file:[{}]'.format(FLAGS.model_path))
    return model


def decode():
    # Load model config
    config = load_config(FLAGS)
    print(config)
    # Load source data to decode
    test_set = TextIterator(source=config['decode_input'],
                            batch_size=config['decode_batch_size'],
                            source_dict=config['source_vocabulary'],
                            n_words_source=config['num_encoder_symbols'])
    
    # Load inverse dictionary used in decoding
    target_inverse_dict = load_inverse_dict(config['target_vocabulary'])
    
    # Initiate TF session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                          log_device_placement=FLAGS.log_device_placement,
                                          gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
        
        # Reload existing checkpoint
        model = load_model(sess, config)
        try:
            print('Decoding {}..'.format(FLAGS.decode_input))
            if FLAGS.write_n_best:
                fout = [open(("%s_%d" % (FLAGS.decode_output, k)), 'w') \
                        for k in range(FLAGS.beam_width)]
            else:
                fout = [open(FLAGS.decode_output, 'w')]
            
            for idx, source_seq in enumerate(test_set.next()):
                print('Source', source_seq)
                source, source_len = prepare_batch(source_seq)
                print('Source', source, 'Source Len', source_len)
                # predicted_ids: GreedyDecoder; [batch_size, max_time_step, 1]
                # BeamSearchDecoder; [batch_size, max_time_step, beam_width]
                predicted_ids = model.predict(sess, encoder_inputs=source,
                                              encoder_inputs_length=source_len)
                print(predicted_ids)
                # Write decoding results
                for k, f in reversed(list(enumerate(fout))):
                    for seq in predicted_ids:
                        f.write(str(seq2words(seq[:, k], target_inverse_dict)) + '\n')
                        f.flush()
                    if not FLAGS.write_n_best:
                        break
                print('{}th line decoded'.format(idx * FLAGS.decode_batch_size))
            
            print('Decoding terminated')
        except IOError:
            pass
        finally:
            [f.close() for f in fout]


def main(_):
    decode()


if __name__ == '__main__':
    tf.app.run()
