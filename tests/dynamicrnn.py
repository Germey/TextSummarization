import tensorflow as tf

batch_size = 32
input_size = 128
time = 10

cell1 = tf.nn.rnn_cell.BasicRNNCell(120)
cell2 = tf.nn.rnn_cell.BasicRNNCell(110)
cell3 = tf.nn.rnn_cell.BasicRNNCell(100)

cells = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2, cell3])
print(cells.state_size)

inputs = tf.placeholder(tf.float32, shape=(batch_size, time, input_size))
print(inputs)

h0 = cells.zero_state(batch_size, tf.float32)

output, hs = tf.nn.dynamic_rnn(cells, inputs, initial_state=h0)

print(output)
print(hs)