from collections import OrderedDict
import tensorflow as tf

tf.app.flags.DEFINE_string('source_vocabulary', 'dataset/nlpcc/articles_vocabs.json', 'Path to source vocabulary')
tf.app.flags.DEFINE_string('target_vocabulary', 'dataset/nlpcc/summaries_vocabs.json', 'Path to target vocabulary')
tf.app.flags.DEFINE_float('learning_rate', 0.0002, 'Learning rate')
FLAGS = tf.app.flags.FLAGS
config = OrderedDict(FLAGS.flag_values_dict())

print(config['learning_rate'])
print(FLAGS.learning_rate)