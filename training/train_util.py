import tensorflow as tf
from transformer_util import *


class CustomLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Creating a custom learning rate schedule that helps faster convergence which is computed by formula:
    d_model^(-0.5) * min( step^(-0.5), step * warmup_steps^(-1.5))
    """

    def __init__(self, d_model, warmup_steps=4000):
        """
        Constructor of the Custom Learning Rate Schedule

        :param d_model: dimension of the word embedding vector
        :param warmup_steps: coefficient in the formula
        """

        super(CustomLearningRateSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        """
        Call function of the Custom Learning Rate Schedule

        :param step: step in the learning rate
        :return: current value of the learning rate to be used
        """

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(tf.math.rsqrt(step), step * (self.warmup_steps ** -1.5))


def loss_function(real, prediction):
    """
    Sparse Categorical Cross Entropy loss function is used for this problem

    :param real: real value
    :param prediction: predicted value
    :return: loss between prediction and real value
    """

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss_ = loss_object(real, prediction)
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


def create_masks(input, target):
    """
    FUnction to create masks to be used in Transformer model

    :param input: input vector
    :param target: target vector
    :return: created masks
    """

    encoder_padding_mask = create_padding_mask(input)
    decoder_padding_mask = create_padding_mask(input)
    look_ahead_mask = create_lookahead_mask(tf.shape(target)[1])
    decoder_target_padding_mask = create_padding_mask(target)
    combined_mask = tf.maximum(decoder_target_padding_mask, look_ahead_mask)
    return encoder_padding_mask, combined_mask, decoder_padding_mask
