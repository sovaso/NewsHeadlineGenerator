import numpy as np
import tensorflow as tf


def get_angles(position, i, d_model):
    """
    Creates an angle that will be used for calculating the positional encoding

    :param position: word position in the text
    :param i: position in the word embedding vector
    :param d_model: dimension of the word embedding vector
    :return: angle for the positional encoding
    """

    return position * (1.0 / np.power(10000, (2 * (i // 2)) / np.float32(d_model)))


def positional_encoding(position, d_model):
    """
    Creating a positional encoding that will be added to the word embedding vector in order to keep the information
    about the position of the word in the text

    :param position: word position in the text
    :param d_model: dimension of the word embedding vector
    :return: positional encoding
    """

    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
    """
    Padding mask for masking "pad" sequences

    :param seq: input sequence
    :return: masked sequence
    """

    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]


def create_lookahead_mask(size):
    """
    Lookahead mask for masking future words from contributing in prediction of current words in self attention

    :param size: size of the requested mask
    :return: mask
    """

    return 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)


def scaled_dot_product_attention(query, key, value, mask):
    """
    Scaled dot product attention is defined as softmax(QK.transpose/sqrt(dk))*V

    :param query: query vector  which is activated and trained when a word vector xn seeks all of the key-value pairs of the
           other word vectors, including itself in self-attention
    :param key: key vector which will be trained to provide an attention value
    :param value: value vector which will be trained to provide another attention value
    :param mask: mask to add to scaled Q and K product
    :return: scaled dot product attention and attention weights
    """

    matmul_qk = tf.matmul(query, key, transpose_b=True)
    dk = tf.cast(tf.shape(key)[-1], tf.float32)
    scaled_qk_product = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_qk_product += (mask * -1e9)
    attention_weights = tf.nn.softmax(scaled_qk_product, axis=-1)
    output = tf.matmul(attention_weights, value)
    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Multi-head attention allows the model to jointly attend to information from different representation subspaces at
    different positions. With a single attention head, averaging inhibits this
    """

    def __init__(self, d_model, number_of_heads):
        """
        Constructor for multi-head attention

        :param d_model: dimension of the word embedding vector
        :param number_of_heads: number of heads to work in parallel
        """

        super(MultiHeadAttention, self).__init__()
        self.number_of_heads = number_of_heads
        self.d_model = d_model
        self.depth = d_model // self.number_of_heads
        self.weights_query = tf.keras.layers.Dense(d_model)
        self.weights_key = tf.keras.layers.Dense(d_model)
        self.weights_value = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """
        Splitting input into heads that will work in parallel

        :param x: word embeddings
        :param batch_size: size of one batch for one head
        :return: reshaped input suitable for heads working in parallel
        """

        x = tf.reshape(x, (batch_size, -1, self.number_of_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, value, key, query, mask):
        """
        Call function for the multi-head attention

        :param query: query vector  which is activated and trained when a word vector xn seeks all of the key-value pairs of the
               other word vectors, including itself in self-attention
        :param key: key vector which will be trained to provide an attention value
        :param value: value vector which will be trained to provide another attention value
        :param mask: mask to add to scaled Q and K product
        :return: output of the multi-head attention and attention weights
        """

        batch_size = tf.shape(query)[0]
        query = self.weights_query(query)
        key = self.weights_key(key)
        value = self.weights_value(value)
        # splitting heads for query, key and value
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        scaled_attention, attention_weights = scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    """
    Classic fully connected feed-forward network which consists of two linear transformations with a ReLU activation in
    between

    :param d_model: dimension of the word embedding vector
    :param dff: inner-layer dimensionality
    :return: output of the point wise feed forward network
    """

    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])
