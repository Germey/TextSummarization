import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np

batch_size = 100
hidden_state = 120

x = np.full((batch_size, hidden_state), False, dtype=np.float32)
print(x)

x0 = chainer.Variable(x)
print(x0.shape)
c = F.broadcast_to(x0, (batch_size, hidden_state))
print(c)

print(c.shape)

lengths = chainer.Variable(np.zeros(shape=(batch_size, 1), dtype=np.float32))

print(lengths.shape)

y = c * lengths
print(y)
