from code.transformer_util import *
import tensorflow as tf


class EncoderLayer(tf.keras.layers.Layer):
    """
    Encoder layer consists of multi-head attention followed by normalization layer and point wised feed forward network
    followed with normalization layer
    """

    def __init__(self, d_model, number_of_heads, dff, rate=0.1):
        """
        Constructor for encoder layer

        :param d_model: dimension of the word embedding vector
        :param number_of_heads: number of heads to work in parallel
        :param dff: inner-layer dimensionality
        :param rate: dropout rate
        """

        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, number_of_heads)
        self.feed_forward_network = point_wise_feed_forward_network(d_model, dff)
        self.normalization_layer1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.normalization_layer2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        """
        Call function for the encoder layer

        :param x: input vector
        :param training: boolean values to do a dropout only when is it a training process
        :param mask: mask for multi-head attention
        :return: output of the encoder layer
        """

        attention_output, _ = self.multi_head_attention(x, x, x, mask)
        attention_output = self.dropout1(attention_output, training=training)
        out1 = self.normalization_layer1(x + attention_output)
        feed_forward_network_output = self.feed_forward_network(out1)
        feed_forward_network_output = self.dropout2(feed_forward_network_output, training=training)
        out2 = self.normalization_layer2(out1 + feed_forward_network_output)
        return out2


class DecoderLayer(tf.keras.layers.Layer):
    """
    Decoder layer consists of masked multi-head attention followed by normalization layer, multi-head attention followed
    by normalization layer and point wise feed forward network followed by normalization layer
    """

    def __init__(self, d_model, number_of_heads, dff, rate=0.1):
        """
        Constructor for decoder layer

        :param d_model: dimension of the word embedding vector
        :param number_of_heads: number of heads to work in parallel
        :param dff: inner-layer dimensionality
        :param rate: dropout rate
        """

        super(DecoderLayer, self).__init__()
        self.masked_multi_head_attention = MultiHeadAttention(d_model, number_of_heads)
        self.multi_head_attention = MultiHeadAttention(d_model, number_of_heads)
        self.feed_forward_network = point_wise_feed_forward_network(d_model, dff)
        self.normalization_layer1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.normalization_layer2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.normalization_layer3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Call function for the decoder layer

        :param x: input vector
        :param encoder_output: output of the encoder
        :param training: boolean values to do a dropout only when is it a training process
        :param look_ahead_mask: look ahead mask for masked multi-head attention
        :param padding_mask: padding mask for multi-head attention
        :return: output of the decoder layer
        """

        attention1, attention_weights_block1 = self.masked_multi_head_attention(x, x, x, look_ahead_mask)
        attention1 = self.dropout1(attention1, training=training)
        out1 = self.normalization_layer1(attention1 + x)
        attention2, attention_weights_block2 = self.multi_head_attention(encoder_output, encoder_output, out1,
                                                                         padding_mask)
        attention2 = self.dropout2(attention2, training=training)
        out2 = self.normalization_layer2(attention2 + out1)
        feed_forward_network_output = self.feed_forward_network(out2)
        feed_forward_network_output = self.dropout3(feed_forward_network_output, training=training)
        out3 = self.normalization_layer3(feed_forward_network_output + out2)
        return out3, attention_weights_block1, attention_weights_block2


class Encoder(tf.keras.layers.Layer):
    """
    Encoder class contains of word embedding, positional encoding and several numbers of encoder layers to represent an
    encoder of transformer model
    """

    def __init__(self, number_of_layers, d_model, number_of_heads, dff, input_vocabulary_size,
                 maximum_position_encoding, rate=0.1):
        """
        Constructor for the Encoder

        :param number_of_layers: number of encoder layers
        :param d_model: dimension of the word embedding vector
        :param number_of_heads: number of heads to work in parallel
        :param dff: inner-layer dimensionality
        :param input_vocabulary_size: size of the input vocabulary
        :param maximum_position_encoding: maximum for positional encoding
        :param rate: dropout rate
        """

        super(Encoder, self).__init__()
        self.d_model = d_model
        self.number_of_layers = number_of_layers
        self.embedding = tf.keras.layers.Embedding(input_vocabulary_size, d_model)
        self.positional_encoding = positional_encoding(maximum_position_encoding, d_model)
        self.encoder_layers = [EncoderLayer(d_model, number_of_heads, dff, rate) for _ in range(number_of_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        """
        Call function of the Encoder

        :param x: input vector
        :param training: boolean values to do a dropout only when is it a training process
        :param mask: mask for multi-head attention for encoder layers
        :return: output vector
        """

        sequence_length = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.positional_encoding[:, :sequence_length, :]
        x = self.dropout(x, training=training)
        for i in range(self.number_of_layers):
            x = self.encoder_layers[i](x, training, mask)
        return x


class Decoder(tf.keras.layers.Layer):
    """
    Decoder class contains of word embedding, positional encoding and several numbers of decoder layers to represent an
    decoder of transformer model
    """

    def __init__(self, number_of_layers, d_model, number_of_heads, dff, target_vocabulary_size,
                 maximum_position_encoding, rate=0.1):
        """
        Constructor for the Decoder

        :param number_of_layers: number of decoder layers
        :param d_model: dimension of the word embedding vector
        :param number_of_heads: number of heads to work in parallel
        :param dff: inner-layer dimensionality
        :param target_vocabulary_size: size of the target vocabulary
        :param maximum_position_encoding: maximum for positional encoding
        :param rate: dropout rate
        """

        super(Decoder, self).__init__()
        self.d_model = d_model
        self.number_of_layers = number_of_layers
        self.embedding = tf.keras.layers.Embedding(target_vocabulary_size, d_model)
        self.positional_encoding = positional_encoding(maximum_position_encoding, d_model)
        self.decoder_layers = [DecoderLayer(d_model, number_of_heads, dff, rate) for _ in range(number_of_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Call function of the Decoder

        :param x: input vector
        :param encoder_output: output of the encoder
        :param training: boolean values to do a dropout only when is it a training process
        :param look_ahead_mask: look ahead mask for masked multi-head attention
        :param padding_mask: padding mask for multi-head attention
        :return: output of the decoder
        """

        sequence_length = tf.shape(x)[1]
        attention_weights = {}
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.positional_encoding[:, :sequence_length, :]
        x = self.dropout(x, training=training)
        for i in range(self.number_of_layers):
            x, block1, block2 = self.decoder_layers[i](x, encoder_output, training, look_ahead_mask, padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2
        return x, attention_weights


class Transformer(tf.keras.Model):
    """
    Transformer class that consist of Encoder class, Decoder class and one final layer
    """

    def __init__(self, number_of_layers, d_model, number_of_heads, dff, input_vocabulary_size, target_vocabulary_size,
                 positional_encoding_input, positional_encoding_target, rate=0.1):
        """
        Constructor of the Transformer

        :param number_of_layers: number of Encoder/Decoder layers
        :param d_model: dimension of the word embedding vector
        :param number_of_heads: number of heads to work in parallel
        :param dff: inner-layer dimensionality
        :param input_vocabulary_size: size of the input vocabulary
        :param target_vocabulary_size: size of the target vocabulary
        :param positional_encoding_input: maximum for input positional encoding
        :param positional_encoding_target: maximum for target positional encoding
        :param rate: dropout rate
        """

        super(Transformer, self).__init__()
        self.encoder = Encoder(number_of_layers, d_model, number_of_heads, dff, input_vocabulary_size,
                               positional_encoding_input, rate)
        self.decoder = Decoder(number_of_layers, d_model, number_of_heads, dff, target_vocabulary_size,
                               positional_encoding_target, rate)
        self.final_layer = tf.keras.layers.Dense(target_vocabulary_size)

    def call(self, input, target, training, encoder_padding_mask, look_ahead_mask, decoder_padding_mask):
        """
        Call function for the Transformer

        :param input: input sequence
        :param target: output sequence
        :param training: boolean values to do a dropout only when is it a training process
        :param encoder_padding_mask: padding mask for encoder
        :param look_ahead_mask: look ahead mask for decoder
        :param decoder_padding_mask: padding mask for decoder
        :return: output of the transformer
        """

        encoder_output = self.encoder(input, training, encoder_padding_mask)
        decoder_output, attention_weights = self.decoder(target, encoder_output, training, look_ahead_mask,
                                                         decoder_padding_mask)
        final_output = self.final_layer(decoder_output)
        return final_output, attention_weights
