import tensorflow as tf

# Adapted from https://www.tensorflow.org/tutorials/text/transformer
def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

        # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class SelfAttention(tf.keras.layers.Layer):
    """
    My implementation of a self attention layer
    """

    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(SelfAttention, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.concat = tf.keras.layers.Concatenate()
        self.sigmoid = tf.keras.layers.Dense(d_model, activation="sigmoid")
        self.linear = tf.keras.layers.Dense(d_model)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x):
        attn_output, _ = self.mha(x, x, x, None)
        attn_output = self.dropout1(attn_output)

        concatenated = self.concat([attn_output, x])
        aoa_output = self.sigmoid(concatenated) * self.linear(concatenated)

        out1 = self.layernorm1(x + aoa_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class GuidedAttention(tf.keras.layers.Layer):
    """
    My implementation of a guided attention layer
    """

    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(GuidedAttention, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.concat = tf.keras.layers.Concatenate()
        self.linear = tf.keras.layers.Dense(d_model)
        self.sigmoid = tf.keras.layers.Dense(d_model, activation="sigmoid")

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, y, x):
        attn_output, _ = self.mha(y, y, x, None)
        attn_output = self.dropout1(attn_output)

        concatenated = self.concat([attn_output, x])
        aoa_output = self.sigmoid(concatenated) * self.linear(concatenated)

        out1 = self.layernorm1(x + aoa_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


class ModularCoAttention(tf.keras.layers.Layer):
    """
    My implementation of a ModularCoAttention stack of self attention and guided
    attention blocks
    """

    def __init__(self, l, d_model, num_heads, dff, rate=0.1):
        super(ModularCoAttention, self).__init__()

        self.encoder = []
        self.decoder_saoa = []
        self.decoder_gaoa = []
        for i in range(0, l):
            self.encoder.append(
                SelfAttention(d_model, num_heads, dff, rate))
            self.decoder_saoa.append(
                SelfAttention(d_model, num_heads, dff, rate))
            self.decoder_gaoa.append(
                GuidedAttention(d_model, num_heads, dff, rate))

    def call(self, y, x):
        out1 = y
        out2 = x
        for saoa in self.encoder:
            out1 = saoa(out1)
        for i in range(0, len(self.decoder_saoa)):
            out2 = self.decoder_saoa[i](out2)
            out2 = self.decoder_gaoa[i](out1, out2)
        return out1, out2


def mlp_network(d):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(d, activation="relu"),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1)
    ])


class MultiModalAttention(tf.keras.layers.Layer):
    """
    My implementation of a question and image feature multi-modal attention block
    """

    def __init__(self, d_model, output_size):
        super(MultiModalAttention, self).__init__()

        self.question_mlp = mlp_network(d_model)
        self.image_mlp = mlp_network(d_model)
        self.question_dot1 = tf.keras.layers.Dot(axes=(1, 1))
        self.image_dot1 = tf.keras.layers.Dot(axes=(1, 1))

        self.concat = tf.keras.layers.Concatenate()
        self.fc1 = tf.keras.layers.Dense(1024)
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.fc2 = tf.keras.layers.Dense(512)
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        self.fc3 = tf.keras.layers.Dense(2, activation="softmax")

        self.question_dot2 = tf.keras.layers.Dot(axes=(1, 1))
        self.image_dot2 = tf.keras.layers.Dot(axes=(1, 1))

        self.sum = tf.keras.layers.Add()

        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.final = tf.keras.layers.Dense(output_size, activation="sigmoid")
        self.reshape = tf.keras.layers.Reshape((output_size,))

    def call(self, y, x):
        y_dash = self.question_dot1([self.question_mlp(y), y])
        x_dash = self.image_dot1([self.image_mlp(x), x])

        middle = self.concat([y_dash, x_dash])
        middle = self.fc1(middle)
        middle = self.dropout1(middle)
        middle = self.fc2(middle)
        middle = self.dropout2(middle)
        middle = self.fc3(middle)

        question_multiplier, image_multiplier = tf.split(middle, num_or_size_splits=2, axis=2)
        a = self.question_dot2([question_multiplier, y_dash])
        b = self.image_dot2([image_multiplier, x_dash])

        out = self.sum([y_dash, x_dash, a, b])
        out = self.norm(out)
        out = self.final(out)
        out = self.reshape(out)
        return out