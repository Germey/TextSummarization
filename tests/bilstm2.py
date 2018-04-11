import tensorflow as tf


def lstm_cell(n_units):
    return tf.nn.rnn_cell.LSTMCell(n_units)


n_units = 100
n_embed = 300
batch_size = 128
time_steps = 20
n_layers = 3

x = tf.Variable(tf.random_normal(shape=[batch_size, time_steps, n_embed], mean=10, stddev=2), dtype=tf.float32)
print('X', x)
inputs = tf.unstack(x, time_steps, axis=1)
print('Inputs', inputs)

cell_fw = [lstm_cell(n_units) for _ in range(n_layers)]
cell_bw = [lstm_cell(n_units) for _ in range(n_layers)]

print(cell_bw, cell_fw)

output, state_fw, state_bw = tf.contrib.rnn.stack_bidirectional_rnn(cell_fw, cell_bw, inputs=inputs, dtype=tf.float32)

output = tf.stack(output, axis=1)

print('Output', output)
print('State Fw', state_fw)
print('State Bw', state_bw)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result_x, result_output = sess.run([x, output])
    # print(result_x)
    # print(result_output)
