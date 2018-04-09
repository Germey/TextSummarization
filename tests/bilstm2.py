import tensorflow as tf


def lstm_cell(n_units):
    return tf.nn.rnn_cell.LSTMCell(n_units)


def multi_cell(cells):
    return tf.nn.rnn_cell.MultiRNNCell(cells)


n_units = 100
n_embed = 300
batch_size = 128
time_steps = 20
n_layers = 3

x = tf.Variable(tf.random_normal(shape=[batch_size, time_steps, n_embed], mean=10, stddev=2), dtype=tf.float32)
print('X', x)

cell_fw = multi_cell([lstm_cell(n_units) for _ in range(n_layers)])
cell_bw = multi_cell([lstm_cell(n_units) for _ in range(n_layers)])

print(cell_bw, cell_fw)

output, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs=x, dtype=tf.float32)

output = tf.concat(output, axis=2)

print('Output', output)
print('State Fw', state_fw)
print('State Bw', state_bw)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result_x, result_output = sess.run([x, output])
    # print(result_x)
    # print(result_output)
