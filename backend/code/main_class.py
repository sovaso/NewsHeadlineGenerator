import re
import time
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import unicodedata
from transformer_model import Transformer
from train_util import *


class NewsHeadlineGenerator:
    """
    Main class for training this project to generate news' headlines
    """

    def __init__(self, number_of_layers, d_model, dff, number_of_heads, epochs, batch_size):
        """
        Constructor for the News Headline Generator

        :param number_of_layers: number of the decoder/encoder layers in transformer
        :param d_model: dimension of the word embedding vector
        :param dff: inner-layer dimensionality
        :param number_of_heads: number of heads to work in parallel
        :param epochs: number of epochs to train
        :param batch_size: batch size for the training a transformer
        """

        self.number_of_layers = number_of_layers
        self.d_model = d_model
        self.dff = dff
        self.number_of_heads = number_of_heads
        self.epochs = epochs
        self.batch_size = batch_size
        self.encoder_max_length = None
        self.decoder_max_length = None
        self.document_train = None
        self.document_test = None
        self.summary_train = None
        self.summary_test = None
        self.encoder_vocabulary_size = None
        self.decoder_vocabulary_size = None
        self.transformer = None
        self.optimizer = None
        self.tokenizer = None
        self.dataset = None
        self.train_loss_log = []
        self.train_accuracy_log = []
        self.test_loss_log = []
        self.test_accuracy_log = []

    def import_train_and_test_set(self, path):
        """
        Importing the dataset in program

        :param path: file path for the dataset
        """

        train_dataset = pd.read_csv(path + "/train.csv")
        self.summary_train = train_dataset["Short"]
        self.summary_train = self.summary_train.apply(lambda row: self.preprocess_sentence(row))
        self.document_train = train_dataset["Headline"]
        self.document_train = self.document_train.apply(lambda row: self.preprocess_sentence(row))
        test_dataset = pd.read_csv(path + "/test.csv")
        self.summary_test = test_dataset["Short"]
        self.summary_test = self.summary_test.apply(lambda row: self.preprocess_sentence(row, is_train=False))
        self.document_test = test_dataset["Headline"]
        self.document_test = self.document_test.apply(lambda row: self.preprocess_sentence(row, is_train=False))

    def preprocess_sentence(self, sentence, is_train=True):
        """
        Preprocessing the sentence in order to make it suitable for the transformer to train on them faster and better

        :param sentence: input sentence
        :param is_train: boolean value to check if it is used for training purposes
        :return: preprocessed output sequence
        """

        def unicode_to_ascii(sequence):
            """
            Converts the unicode file to ascii

            :param sequence: unicode input sequence
            :return: ascii output sequence
            """

            return ''.join(c for c in unicodedata.normalize('NFD', sequence) if unicodedata.category(c) != 'Mn')

        sentence = unicode_to_ascii(sentence.lower().strip())
        if is_train:
            sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
            sentence = re.sub(r'[" "]+', " ", sentence)
            sentence = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sentence)
        sentence = sentence.strip()
        sentence = unicode_to_ascii('<go> ') + sentence + unicode_to_ascii(' <stop>')
        return sentence

    def preprocess_train_and_test_set(self):
        """
        Function used to calculate needed parameters for training the transformer model, while also preprocessing given
        train and test datasets
        """

        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
        self.tokenizer.fit_on_texts(self.summary_train)
        self.encoder_vocabulary_size = len(self.tokenizer.word_index) + 1
        self.decoder_vocabulary_size = len(self.tokenizer.word_index) + 1
        self.encoder_max_length = 89
        self.decoder_max_length = 20
        self.summary_train = self.tokenizer.texts_to_sequences(self.summary_train)
        self.summary_train = tf.keras.preprocessing.sequence.pad_sequences(self.summary_train, padding='post',
                                                                           maxlen=self.encoder_max_length)
        self.document_train = self.tokenizer.texts_to_sequences(self.document_train)
        self.document_train = tf.keras.preprocessing.sequence.pad_sequences(self.document_train, padding='post',
                                                                            maxlen=self.decoder_max_length)
        self.summary_test = self.tokenizer.texts_to_sequences(self.summary_test)
        self.summary_test = tf.keras.preprocessing.sequence.pad_sequences(self.summary_test, padding='post',
                                                                          maxlen=self.encoder_max_length)
        self.document_test = self.tokenizer.texts_to_sequences(self.document_test)
        self.document_test = tf.keras.preprocessing.sequence.pad_sequences(self.document_test, padding='post',
                                                                           maxlen=self.decoder_max_length)
        self.dataset = tf.data.Dataset.from_tensor_slices((self.summary_train, self.document_train)).shuffle(
            len(self.summary_train))
        self.dataset = self.dataset.padded_batch(self.batch_size, drop_remainder=True)
        self.dataset = self.dataset.prefetch(tf.data.experimental.AUTOTUNE)

    def writing_loss_and_accuracy_values_to_file(self):
        """
        Writing loss and accuracy to the txt file for both test and train dataset
        """

        path = "../log_files"
        with open(path + "/train_loss_log.txt", 'w+') as fp:
            fp.write(str(self.train_loss_log))
        with open(path + "/test_loss_log.txt", 'w+') as fp:
            fp.write(str(self.test_loss_log))
        with open(path + "/train_accuracy_log.txt", 'w+') as fp:
            fp.write(str(self.train_accuracy_log))
        with open(path + "/test_accuracy_log.txt", 'w+') as fp:
            fp.write(str(self.test_accuracy_log))

    def train_transformer(self, is_train=True):
        """
        Main function for training the transformer

        :param is_train: boolean value to check if it is called for training purposes
        """

        self.preprocess_train_and_test_set()
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

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

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
            train_accuracy(target_real, predictions)

        if is_train:
            for epoch in range(self.epochs):
                start_time = time.time()
                train_loss.reset_states()
                train_accuracy.reset_states()
                test_loss.reset_states()
                test_accuracy.reset_states()
                for (batch, (input, target)) in enumerate(self.dataset):
                    train_step(input, target)
                    if batch % 10 == 0:
                        print('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, batch,
                                                                                     train_loss.result(),
                                                                                     train_accuracy.result()))
                if (epoch + 1) % 5 == 0:
                    checkpoint_save_path = checkpoint_manager.save()
                    print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, checkpoint_save_path))
                    self.writing_loss_and_accuracy_values_to_file()
                print('[TRAIN] Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, train_loss.result(),
                                                                    train_accuracy.result()))
                time_needed_for_epoch = time.time() - start_time
                test_target_input = self.document_test[:, :-1]
                test_target_real = self.document_test[:, 1:]
                encoder_padding_mask, combined_mask, decoder_padding_mask = create_masks(self.summary_test,
                                                                                         test_target_input)
                test_predictions, _ = self.transformer(self.summary_test, test_target_input, training=True,
                                                       encoder_padding_mask=encoder_padding_mask,
                                                       look_ahead_mask=combined_mask,
                                                       decoder_padding_mask=decoder_padding_mask)
                loss = loss_function(test_target_real, test_predictions)
                test_loss(loss)
                test_accuracy(test_target_real, test_predictions)
                print('[TEST] Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, test_loss.result(),
                      test_accuracy.result()))
                print('Time taken for this epoch: {} secs\n'.format(time_needed_for_epoch))
                # writing loss and accuracy from this epoch in lists
                self.train_loss_log.append(train_loss.result())
                self.train_accuracy_log.append(train_accuracy.result())
                self.test_loss_log.append(test_loss.result())
                self.test_accuracy_log.append(test_accuracy.result())

    def test_transformer(self, input_sequence):
        """
        Function to test the trained transformer to see how it is performing

        :param input_sequence: input sequence (short)
        :return: predicted output sequence (headline)
        """

        self.train_transformer(is_train=False)
        input_sequence = self.preprocess_sentence(input_sequence)
        input_sequence = self.tokenizer.texts_to_sequences([input_sequence])
        input_sequence = tf.keras.preprocessing.sequence.pad_sequences(input_sequence, maxlen=self.encoder_max_length,
                                                                       padding='post', truncating='post')
        encoder_input = tf.expand_dims(input_sequence[0], 0)
        decoder_input = [self.tokenizer.word_index["<go>"]]
        output = tf.expand_dims(decoder_input, 0)
        for i in range(self.decoder_max_length):
            enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, output)
            predictions, attention_weights = self.transformer(encoder_input, output, False, enc_padding_mask,
                                                              combined_mask, dec_padding_mask)
            predictions = predictions[:, -1:, :]
            predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            if predicted_id == self.tokenizer.word_index["<stop>"]:
                break
            output = tf.concat([output, predicted_id], axis=-1)
        summarized = tf.squeeze(output, axis=0)
        summarized = np.expand_dims(summarized[1:], 0)
        return self.tokenizer.sequences_to_texts(summarized)[0]
