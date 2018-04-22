from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
import tensorflow as tf

# TODO(ebrevdo): Remove once _linear is fully deprecated.
linear = core_rnn_cell_impl._linear  # pylint: disable=protected-access


def attention_decoder(decoder_inputs,
                      initial_state,
                      attention_states,
                      cell,
                      output_size=None,
                      num_heads=1,
                      loop_function=None,
                      dtype=None,
                      scope=None,
                      initial_state_attention=False):
    """RNN decoder with attention for the sequence-to-sequence model.

      In this context "attention" means that, during decoding, the RNN can look up
      information in the additional tensor attention_states, and it does this by
      focusing on a few entries from the tensor. This model has proven to yield
      especially good results in a number of sequence-to-sequence tasks. This
      implementation is based on http://arxiv.org/abs/1412.7449 (see below for
      details). It is recommended for complex sequence-to-sequence tasks.

      Args:
        decoder_inputs: A list of 2D Tensors [batch_size x input_size].
        initial_state: 2D Tensor [batch_size x cell.state_size].
        attention_states: 3D Tensor [batch_size x attn_length x attn_size].
        cell: core_rnn_cell.RNNCell defining the cell function and size.
        output_size: Size of the output vectors; if None, we use cell.output_size.
        num_heads: Number of attention heads that read from attention_states.
        loop_function: If not None, this function will be applied to i-th output
          in order to generate i+1-th input, and decoder_inputs will be ignored,
          except for the first element ("GO" symbol). This can be used for decoding,
          but also for training to emulate http://arxiv.org/abs/1506.03099.
          Signature -- loop_function(prev, i) = next
            * prev is a 2D Tensor of shape [batch_size x output_size],
            * i is an integer, the step number (when advanced control is needed),
            * next is a 2D Tensor of shape [batch_size x input_size].
        dtype: The dtype to use for the RNN initial state (default: tf.float32).
        scope: VariableScope for the created subgraph; default: "attention_decoder".
        initial_state_attention: If False (default), initial attentions are zero.
          If True, initialize the attentions from the initial state and attention
          states -- useful when we wish to resume decoding from a previously
          stored decoder state and attention states.

      Returns:
        A tuple of the form (outputs, state), where:
          outputs: A list of the same length as decoder_inputs of 2D Tensors of
            shape [batch_size x output_size]. These represent the generated outputs.
            Output i is computed from input i (which is either the i-th element
            of decoder_inputs or loop_function(output {i-1}, i)) as follows.
            First, we run the cell on a combination of the input and previous
            attention masks:
              cell_output, new_state = cell(linear(input, prev_attn), prev_state).
            Then, we calculate new attention masks:
              new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
            and then we calculate the output:
              output = linear(cell_output, new_attn).
          state: The state of each decoder cell the final time-step.
            It is a 2D Tensor of shape [batch_size x cell.state_size].

      Raises:
        ValueError: when num_heads is not positive, there are no inputs, shapes
          of attention_states are not set, or input size cannot be inferred
          from the input.
    """
    if not decoder_inputs:
        raise ValueError("Must provide at least 1 input to attention decoder.")
    if num_heads < 1:
        raise ValueError("With less than 1 heads, use a non-attention decoder.")
    if attention_states.get_shape()[2].value is None:
        raise ValueError("Shape[2] of attention_states must be known: %s" %
                         attention_states.get_shape())
    if output_size is None:
        output_size = cell.output_size
    
    with variable_scope.variable_scope(
            scope or "attention_decoder", dtype=dtype) as scope:
        dtype = scope.dtype
        
        batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.
        attn_length = attention_states.get_shape()[1].value
        if attn_length is None:
            attn_length = array_ops.shape(attention_states)[1]
        attn_size = attention_states.get_shape()[2].value
        
        # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
        hidden = array_ops.reshape(attention_states,
                                   [-1, attn_length, 1, attn_size])
        hidden_features = []
        v = []
        attention_vec_size = attn_size  # Size of query vectors for attention.
        for a in range(num_heads):
            k = variable_scope.get_variable("AttnW_%d" % a,
                                            [1, 1, attn_size, attention_vec_size])
            hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
            v.append(
                variable_scope.get_variable("AttnV_%d" % a, [attention_vec_size]))
        
        state = initial_state
        
        def attention(query):
            """Put attention masks on hidden using hidden_features and query."""
            ds = []  # Results of attention reads will be stored here.
            if nest.is_sequence(query):  # If the query is a tuple, flatten it.
                query_list = nest.flatten(query)
                for q in query_list:  # Check that ndims == 2 if specified.
                    ndims = q.get_shape().ndims
                    if ndims:
                        assert ndims == 2
                query = array_ops.concat(query_list, 1)
            for a in range(num_heads):
                with variable_scope.variable_scope("Attention_%d" % a):
                    y = linear(query, attention_vec_size, True)
                    y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
                    # Attention mask is a softmax of v^T * tanh(...).
                    s = math_ops.reduce_sum(v[a] * math_ops.tanh(hidden_features[a] + y),
                                            [2, 3])
                    a = nn_ops.softmax(s)
                    # Now calculate the attention-weighted vector d.
                    d = math_ops.reduce_sum(
                        array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden, [1, 2])
                    ds.append(array_ops.reshape(d, [-1, attn_size]))
            return ds
        
        outputs = []
        prev = None
        batch_attn_size = array_ops.stack([batch_size, attn_size])
        attns = [
            array_ops.zeros(
                batch_attn_size, dtype=dtype) for _ in range(num_heads)
        ]
        for a in attns:  # Ensure the second shape of attention vectors is set.
            a.set_shape([None, attn_size])
        if initial_state_attention:
            attns = attention(initial_state)
        for i, inp in enumerate(decoder_inputs):
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
            # If loop_function is set, we use it instead of decoder_inputs.
            if loop_function is not None and prev is not None:
                with variable_scope.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, i)
            # Merge input and previous attentions into one vector of the right size.
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError("Could not infer input size from input: %s" % inp.name)
            x = linear([inp] + attns, input_size, True)
            # Run the RNN.
            cell_output, state = cell(x, state)
            # Run the attention mechanism.
            if i == 0 and initial_state_attention:
                with variable_scope.variable_scope(
                    variable_scope.get_variable_scope(), reuse=True):
                    attns = attention(state)
            else:
                attns = attention(state)
            
            with variable_scope.variable_scope("AttnOutputProjection"):
                output = linear([cell_output] + attns, output_size, True)
            if loop_function is not None:
                prev = output
            outputs.append(output)
    
    return outputs, state


def double_attention_decoder(decoder_inputs,
                             initial_state,
                             attention_states,
                             cell,
                             source_emb_encoder_inputs,
                             output_size=None,
                             num_heads=1,
                             loop_function=None,
                             dtype=None,
                             scope=None,
                             initial_state_attention=False):
    """RNN decoder with attention for the sequence-to-sequence model.

      In this context "attention" means that, during decoding, the RNN can look up
      information in the additional tensor attention_states, and it does this by
      focusing on a few entries from the tensor. This model has proven to yield
      especially good results in a number of sequence-to-sequence tasks. This
      implementation is based on http://arxiv.org/abs/1412.7449 (see below for
      details). It is recommended for complex sequence-to-sequence tasks.

      Args:
        decoder_inputs: A list of 2D Tensors [batch_size x input_size].
        initial_state: 2D Tensor [batch_size x cell.state_size].
        attention_states: 3D Tensor [batch_size x attn_length x attn_size].
        cell: core_rnn_cell.RNNCell defining the cell function and size.
        output_size: Size of the output vectors; if None, we use cell.output_size.
        num_heads: Number of attention heads that read from attention_states.
        loop_function: If not None, this function will be applied to i-th output
          in order to generate i+1-th input, and decoder_inputs will be ignored,
          except for the first element ("GO" symbol). This can be used for decoding,
          but also for training to emulate http://arxiv.org/abs/1506.03099.
          Signature -- loop_function(prev, i) = next
            * prev is a 2D Tensor of shape [batch_size x output_size],
            * i is an integer, the step number (when advanced control is needed),
            * next is a 2D Tensor of shape [batch_size x input_size].
        dtype: The dtype to use for the RNN initial state (default: tf.float32).
        scope: VariableScope for the created subgraph; default: "attention_decoder".
        initial_state_attention: If False (default), initial attentions are zero.
          If True, initialize the attentions from the initial state and attention
          states -- useful when we wish to resume decoding from a previously
          stored decoder state and attention states.

      Returns:
        A tuple of the form (outputs, state), where:
          outputs: A list of the same length as decoder_inputs of 2D Tensors of
            shape [batch_size x output_size]. These represent the generated outputs.
            Output i is computed from input i (which is either the i-th element
            of decoder_inputs or loop_function(output {i-1}, i)) as follows.
            First, we run the cell on a combination of the input and previous
            attention masks:
              cell_output, new_state = cell(linear(input, prev_attn), prev_state).
            Then, we calculate new attention masks:
              new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
            and then we calculate the output:
              output = linear(cell_output, new_attn).
          state: The state of each decoder cell the final time-step.
            It is a 2D Tensor of shape [batch_size x cell.state_size].

      Raises:
        ValueError: when num_heads is not positive, there are no inputs, shapes
          of attention_states are not set, or input size cannot be inferred
          from the input.
    """
    if not decoder_inputs:
        raise ValueError("Must provide at least 1 input to attention decoder.")
    if num_heads < 1:
        raise ValueError("With less than 1 heads, use a non-attention decoder.")
    if attention_states.get_shape()[2].value is None:
        raise ValueError("Shape[2] of attention_states must be known: %s" %
                         attention_states.get_shape())
    if output_size is None:
        output_size = cell.output_size
    
    with variable_scope.variable_scope(
            scope or "attention_decoder", dtype=dtype) as scope:
        dtype = scope.dtype
        # batch_size:inputsize
        batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.
        # 得到enc_timesteps
        attn_length = attention_states.get_shape()[1].value
        if attn_length is None:
            attn_length = array_ops.shape(attention_states)[1]
        # 2*num_hidden
        attn_size = attention_states.get_shape()[2].value
        
        # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
        hidden = array_ops.reshape(attention_states,
                                   [-1, attn_length, 1, attn_size])
        
        
        hidden_features = []
        v = []
        de_v = []
        attention_vec_size = attn_size  # Size of query vectors for attention.
        for a in range(num_heads):
            k = variable_scope.get_variable("AttnW_%d" % a,
                                            [1, 1, attn_size, attention_vec_size])
            # 用二维卷积来得到W1*h_t来处理attention_states
            hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
            # attention的参数
            v.append(
                variable_scope.get_variable("AttnV_%d" % a, [attention_vec_size]))
            
            de_v.append(
                variable_scope.get_variable("AttnV_de_%d" % a, [cell.state_size]))
        state = initial_state
        
        # query代表decoder的state
        # hidden_features是经过二维卷积处理过的encoder的state
        # ds == Ct
        
        #         sigmoidWeight = tf.get_variable('sigmoidWeight', [attention_vec_size],dtype=tf.float32)
        #         sigmoidBiase = tf.get_variable('sigmoidBiase' , [attention_vec_size],dtype=tf.float32)
        
        def attention(query, decoder_state):
            """Put attention masks on hidden using hidden_features and query."""
            ds = []  # Results of attention reads will be stored here.
            de_ds = []
            if nest.is_sequence(query):  # If the query is a tuple, flatten it.
                query_list = nest.flatten(query)
                for q in query_list:  # Check that ndims == 2 if specified.
                    ndims = q.get_shape().ndims
                    if ndims:
                        assert ndims == 2
                query = array_ops.concat(query_list, 1)
            for a in range(num_heads):
                with variable_scope.variable_scope("Attention_encode_%d" % a):
                    # hidden_features是4维[-1, attn_length, 1, attn_size]，所以将query变为4维
                    y = linear(query, attention_vec_size, True)
                    y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
                    # F(hj,Hi):Attention mask is a softmax of v^T * tanh(...).
                    s = math_ops.reduce_sum(v[a] * math_ops.tanh(hidden_features[a] + y),
                                            [2, 3])
                    # attention系数[attn_length]
                    e_a = nn_ops.softmax(s)

                    # a和de_a合起来对encoder的hj进行attention
                    # Now calculate the attention-weighted vector d.
                    # aij*hj
                    # hidden + source_emb_encoder_inputs:将state和源语言嵌入相加
                    #                     hidden_source = hidden + source_emb_encoder_inputs
                    hidden_source = hidden
                    d = math_ops.reduce_sum(array_ops.reshape(e_a, [-1, attn_length, 1, 1]) * hidden_source,
                                            [1, 2])
                    
                    ds.append(array_ops.reshape(d, [-1, attn_size]))
                
                with variable_scope.variable_scope("Attention_decode_%d" % a):
                    # 得出decoder的de_s.要把query的维度改变成需要的
                    de_y = linear(query, cell.state_size, True)  # [batchsize x cell.state_size]
                    de_y = array_ops.reshape(de_y, [-1, 1, cell.state_size])
                    decoder_state = array_ops.reshape(decoder_state, [batch_size, len(decoder_state), cell.state_size])
                    # F(hj,Hi):Attention mask is a softmax of v^T * tanh(...).
                    
                    de_s = math_ops.reduce_sum(de_v[a] * math_ops.tanh(decoder_state + de_y),
                                               [2])

                    # decoder_attention系数[dec_length]不定长
                    de_a = nn_ops.softmax(de_s)

                    # a和de_a合起来对encoder的hj进行attention
                    # Now calculate the attention-weighted vector d.
                    # aij*hj
                    
                    de_d = math_ops.reduce_sum(
                        array_ops.reshape(de_a, [-1, tf.shape(decoder_state)[1], 1]) * decoder_state,
                        [1])
                    de_ds.append(array_ops.reshape(de_d, [-1, cell.state_size]))
            
            return (ds, de_ds)
        
        outputs = []
        prev = None
        # 刚开始decoderstate还没有就为0
        decoderStateList = []
        initstate = tf.zeros_like(state, tf.float32)
        decoderStateList.append(initstate)

        # batch_attn_size = shape[batch_size*attn_size]
        batch_attn_size = array_ops.stack([batch_size, attn_size])
        de_batch_attn_size = array_ops.stack([batch_size, cell.state_size])

        # attns初始化为0
        attns = [
            array_ops.zeros(
                batch_attn_size, dtype=dtype) for _ in range(num_heads)
        ]
        
        de_attns = [
            array_ops.zeros(
                de_batch_attn_size, dtype=dtype) for _ in range(num_heads)
        ]
        
        for a in attns:  # Ensure the second shape of attention vectors is set.
            a.set_shape([None, attn_size])
        
        for d_a in de_attns:  # Ensure the second shape of attention vectors is set.
            d_a.set_shape([None, cell.state_size])
        
        if initial_state_attention:  # if true,use the encoder fw_end_state initial the attns
            print("initial_state_attention")
            attns, de_attns = attention(initial_state, decoderStateList)
        
        for i, inp in enumerate(decoder_inputs):
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
            # If loop_function is set, we use it instead of decoder_inputs.
            if loop_function is not None and prev is not None:
                with variable_scope.variable_scope("loop_function", reuse=True):
                    inp = loop_function(prev, i)
            # Merge input and previous attentions into one vector of the right size.
            # 将inp和attention投影到一个向量空间中
            input_size = inp.get_shape().with_rank(2)[1]
            if input_size.value is None:
                raise ValueError("Could not infer input size from input: %s" % inp.name)
            # RNN cell输入，对decoderinput做一个线性变换，t-1时刻y加上attention机制得到Yt和St
            # copy机制要对x进行改变
            x = linear([inp] + attns + de_attns, input_size, True)
            # Run the RNN.
            cell_output, state = cell(x, state)
            decoderStateList.append(state)
            # Run the attention mechanism.将atten机制和cell_output合起来得到新的输出
            if i == 0 and initial_state_attention:
                with variable_scope.variable_scope(
                    variable_scope.get_variable_scope(), reuse=True):
                    attns, de_attns = attention(state, decoderStateList)
            else:
                attns, de_attns = attention(state, decoderStateList)
            
            with variable_scope.variable_scope("AttnOutputProjection"):
                output = linear([cell_output] + attns + de_attns, output_size, True)
            
            if loop_function is not None:
                prev = output
            outputs.append(output)
    
    return outputs, state
