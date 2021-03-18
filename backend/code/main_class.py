import re
import time
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from code.transformer_model import Transformer
from code.train_util import *


class NewsHeadlineGenerator:
    """
    Main class for training this project to generate news' headlines
    """

    def __init__(self, number_of_layers, d_model, dff, number_of_heads, epochs, encoder_max_length, decoder_max_length,
                 buffer_size, batch_size):
        """
        Constructor for the News Headline Generator

        :param number_of_layers: number of the decoder/encoder layers in transformer
        :param d_model: dimension of the word embedding vector
        :param dff: inner-layer dimensionality
        :param number_of_heads: number of heads to work in parallel
        :param epochs: number of epochs to train
        :param encoder_max_length: maximal length of the input
        :param decoder_max_length: maximal length of the target
        :param buffer_size: buffer size for the shuffle of the dataset
        :param batch_size: batch size for the training a transformer
        """

        self.number_of_layers = number_of_layers
        self.d_model = d_model
        self.dff = dff
        self.number_of_heads = number_of_heads
        self.epochs = epochs
        self.encoder_max_length = encoder_max_length
        self.decoder_max_length = decoder_max_length
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.document = None
        self.summary = None
        self.encoder_vocabulary_size = None
        self.decoder_vocabulary_size = None
        self.dataset = None
        self.transformer = None
        self.optimizer = None
        self.summary_tokenizer = None
        self.document_tokenizer = None

    def import_dataset(self, path, document_column_name, summary_column_name):
        """
        Importing the dataset in program

        :param path: file path for the dataset
        :param document_column_name: name of the document column in set dataframe
        :param summary_column_name: name of the summary column in set dataframe
        """

        dataset = pd.read_excel(path)
        self.document = dataset[document_column_name]
        self.summary = dataset[summary_column_name]

    def preprocess_data(self):
        """
        Preprocessing the data in order to make it suitable for the transformer to train on them faster and better
        """

        # replacing &#39; sequence of characters to represent '
        self.document = self.document.str.replace('&#39;', '\'')
        self.summary = self.summary.str.replace('&#39;', '\'')
        # for recognizing the start and end of target (summary) sequences, we pad them with start (“<go>”) and end
        # (“<stop>”) tokens
        self.summary = self.summary.apply(lambda x: '<go> ' + x + ' <stop>')
        # filtering unnecessary characters from the input and target
        filters = '!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n'
        oov_token = '<unk>'
        self.document_tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token=oov_token)
        self.summary_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters=filters, oov_token=oov_token)
        self.document_tokenizer.fit_on_texts(self.document)
        self.summary_tokenizer.fit_on_texts(self.summary)
        inputs = self.document_tokenizer.texts_to_sequences(self.document)
        targets = self.summary_tokenizer.texts_to_sequences(self.summary)
        self.encoder_vocabulary_size = len(self.document_tokenizer.word_index) + 1
        self.decoder_vocabulary_size = len(self.summary_tokenizer.word_index) + 1
        inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=self.encoder_max_length, padding='post',
                                                               truncating='post')
        targets = tf.keras.preprocessing.sequence.pad_sequences(targets, maxlen=self.decoder_max_length, padding='post',
                                                                truncating='post')
        inputs = tf.cast(inputs, dtype=tf.int32)
        targets = tf.cast(targets, dtype=tf.int32)
        self.dataset = tf.data.Dataset.from_tensor_slices((inputs, targets)).shuffle(self.buffer_size).batch(
                                                                                                    self.batch_size)

    def train_transformer(self, train=True):
        """
        Main function for training the transformer
        """

        learning_rate = CustomLearningRateSchedule(self.d_model)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.transformer = Transformer(self.number_of_layers, self.d_model, self.number_of_heads, self.dff,
                                       self.encoder_vocabulary_size, self.decoder_vocabulary_size,
                                       positional_encoding_input=self.encoder_vocabulary_size,
                                       positional_encoding_target=self.decoder_vocabulary_size)
        # making the checkpoints to keep the parameters for the model after some epochs
        checkpoint_path = "../checkpoints"
        checkpoint = tf.train.Checkpoint(transformer=self.transformer, optimizer=self.optimizer)
        checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=2)
        if checkpoint_manager.latest_checkpoint:
            checkpoint.restore(checkpoint_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')

        @tf.function
        def train_step(input, target):
            """
            Train step function for the transformer to call the transformer, calculate the loss and update weights of
            the transformer model

            :param input: input value
            :param target: target value
            """

            target_input = target[:, :-1]
            target_real = target[:, 1:]
            encoder_padding_mask, combined_mask, decoder_padding_mask = create_masks(input, target_input)
            with tf.GradientTape() as tape:
                predictions, _ = self.transformer(input, target_input, training=True,
                                                  encoder_padding_mask=encoder_padding_mask,
                                                  look_ahead_mask=combined_mask,
                                                  decoder_padding_mask=decoder_padding_mask)
                loss = loss_function(target_real, predictions)
            gradients = tape.gradient(loss, self.transformer.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.transformer.trainable_variables))
            train_loss(loss)

        if train:
            for epoch in range(self.epochs):
                start_time = time.time()
                train_loss.reset_states()
                for (batch, (input, target)) in enumerate(self.dataset):
                    train_step(input, target)
                    if batch % 10 == 0:
                        print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, train_loss.result()))
                if (epoch + 1) % 5 == 0:
                    checkpoint_save_path = checkpoint_manager.save()
                    print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, checkpoint_save_path))
                print('Epoch {} Loss {:.4f}'.format(epoch + 1, train_loss.result()))
                print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start_time))

    def test_transformer(self, input_document):
        self.train_transformer(train=False)
        input_document = self.document_tokenizer.texts_to_sequences([input_document])
        input_document = tf.keras.preprocessing.sequence.pad_sequences(input_document, maxlen=self.encoder_max_length,
                                                                       padding='post', truncating='post')
        encoder_input = tf.expand_dims(input_document[0], 0)

        decoder_input = [self.summary_tokenizer.word_index["<go>"]]
        output = tf.expand_dims(decoder_input, 0)
        summarized = None
        for i in range(self.decoder_max_length):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)

            predictions, attention_weights = self.transformer(
                encoder_input,
                output,
                False,
                enc_padding_mask,
                combined_mask,
                dec_padding_mask
            )

            predictions = predictions[:, -1:, :]
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

            if predicted_id == self.summary_tokenizer.word_index["<stop>"]:
                break

            output = tf.concat([output, predicted_id], axis=-1)

        summarized = tf.squeeze(output, axis=0)
        summarized = np.expand_dims(summarized[1:], 0)
        return self.summary_tokenizer.sequences_to_texts(summarized)[0]
