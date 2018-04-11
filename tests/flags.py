import tensorflow as tf

tf.app.flags.DEFINE_integer('rnn_size', 128, 'Number of hidden units in each layer')
tf.app.flags.DEFINE_integer('batch_size', 100, 'Number of batch size')
tf.app.flags.DEFINE_integer('time_steps', 30, 'Number of time steps')
tf.app.flags.DEFINE_integer('layer_size', 3, 'Number of layers in each encoder and decoder')
tf.app.flags.DEFINE_integer('embedding_dim', 300, 'Embedding dimensions of encoder and decoder inputs')
tf.app.flags.DEFINE_integer('encoder_vocab_size', 21548, 'Source vocabulary size')
tf.app.flags.DEFINE_integer('decoder_vocab_size', 21548, 'Target vocabulary size')
tf.app.flags.DEFINE_float('grad_clip', 0.01, 'Grid Clip')
tf.app.flags.DEFINE_boolean('is_inference', False, 'Is Influence')

FLAGS = tf.app.flags.FLAGS
config = FLAGS.flag_values_dict()
