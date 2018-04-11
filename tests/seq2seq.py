import tensorflow as tf
from tensorflow.contrib.seq2seq import *
from tensorflow.python.layers.core import Dense


class Seq2SeqModel(object):
    def __init__(self, rnn_size, layer_size, batch_size, time_steps, encoder_vocab_size,
                 decoder_vocab_size, embedding_dim, grad_clip, is_inference=False):
        # define inputs
        self.input_x = tf.placeholder(tf.int32, shape=[batch_size, time_steps], name='input_ids')
        print('Input X', self.input_x)
        # define embedding layer
        with tf.variable_scope('embedding'):
            encoder_embedding = tf.Variable(tf.truncated_normal(shape=[encoder_vocab_size, embedding_dim], stddev=0.1),
                                            name='encoder_embedding')
            decoder_embedding = tf.Variable(tf.truncated_normal(shape=[decoder_vocab_size, embedding_dim], stddev=0.1),
                                            name='decoder_embedding')
        
        print('Encoder Embedding', encoder_embedding)
        print('Decoder Embedding', decoder_embedding)
        
        # define encoder
        with tf.variable_scope('encoder'):
            encoder = self._get_simple_lstm(rnn_size, layer_size)
        
        print('Encoder', encoder)
        
        with tf.device('/cpu:0'):
            input_x_embedded = tf.nn.embedding_lookup(encoder_embedding, self.input_x)
        
        print('Input X Embeded', input_x_embedded)
        
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder, input_x_embedded, dtype=tf.float32)
        
        print('Encoder Outputs', encoder_outputs)
        print('Encoder State', encoder_state)
        
        # define helper for decoder
        if is_inference:
            self.start_tokens = tf.placeholder(tf.int32, shape=[None], name='start_tokens')
            self.end_token = tf.placeholder(tf.int32, name='end_token')
            helper = GreedyEmbeddingHelper(decoder_embedding, self.start_tokens, self.end_token)
        else:
            self.target_ids = tf.placeholder(tf.int32, shape=[batch_size, time_steps], name='target_ids')
            self.decoder_seq_length = tf.placeholder(tf.int32, shape=[batch_size], name='batch_seq_length')
            with tf.device('/cpu:0'):
                target_embeddeds = tf.nn.embedding_lookup(decoder_embedding, self.target_ids)
            
            print('Target Embededs', target_embeddeds)
            
            helper = TrainingHelper(target_embeddeds, self.decoder_seq_length)
            print(helper)
        
        with tf.variable_scope('decoder'):
            fc_layer = Dense(decoder_vocab_size)
            decoder_cell = self._get_simple_lstm(rnn_size, layer_size)
            decoder = BasicDecoder(decoder_cell, helper, encoder_state, fc_layer)
            
            print('Decoder', decoder)
        
        logits, final_state, final_sequence_lengths = dynamic_decode(decoder)
        
        print('Logits', logits)
        print('Final State', final_state)
        print('Final Squence Length', final_sequence_lengths)
        
        if not is_inference:
            targets = tf.reshape(self.target_ids, [-1])
            print('Targets', targets)
            
            logits_flat = tf.reshape(logits.rnn_output, [-1, decoder_vocab_size])
            print('shape logits_flat:{}'.format(logits_flat.shape))
            print('shape logits:{}'.format(logits.rnn_output.shape))
            
            self.cost = tf.losses.sparse_softmax_cross_entropy(targets, logits_flat)
            
            print('Cost', self.cost)
            
            # define train op
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), grad_clip)
            
            optimizer = tf.train.AdamOptimizer(1e-3)
            print('Optimizer', optimizer)
            
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))
            print('Train Op', self.train_op)
        else:
            self.prob = tf.nn.softmax(logits)
    
    def _get_simple_lstm(self, rnn_size, layer_size):
        lstm_layers = [tf.contrib.rnn.LSTMCell(rnn_size) for _ in range(layer_size)]
        return tf.contrib.rnn.MultiRNNCell(lstm_layers)


from flags import config

print(config)

model = Seq2SeqModel(**config)
