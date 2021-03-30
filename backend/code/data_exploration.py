import pandas as pd
import numpy as np
import unicodedata
import re


def preprocess_sentence(sentence, is_train=True):
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

if __name__ == "__main__":
    """
    Main function for the data exploration of the train set
    """

    # read train and test set
    train_dataset = pd.read_csv("../dataset/train.csv")
    test_dataset = pd.read_csv("../dataset/test.csv")

    # dropping the irrelevant columns from the dataframe
    train_dataset.drop(['Source ', 'Time ', 'Publish Date'], axis=1, inplace=True)
    test_dataset.drop(['Source ', 'Time ', 'Publish Date'], axis=1, inplace=True)

    # preprocess train and test dataset
    summary_train = train_dataset["Short"]
    summary_train = summary_train.apply(lambda row: preprocess_sentence(row))
    document_train = train_dataset["Headline"]
    document_train = document_train.apply(lambda row: preprocess_sentence(row))

    summary_test = test_dataset["Short"]
    summary_test = summary_test.apply(lambda row: preprocess_sentence(row, is_train=False))
    document_test = test_dataset["Headline"]
    document_test = document_test.apply(lambda row: preprocess_sentence(row, is_train=False))

    # printing the head of train and test datasets
    print("=== SUMMARY TRAIN DATASET HEAD ===")
    print(summary_train.head())
    print("=== DOCUMENT TRAIN DATASET HEAD ===")
    print(document_train.head())
    print("=== SUMMARY TEST DATASET HEAD ===")
    print(summary_test.head())
    print("=== DATASET TEST DATASET HEAD ===")
    print(document_test.head())

    # printing the length of the sequence for both train and test datasets
    max_length_summary_train = 0
    for i in range(len(summary_train)):
        if max_length_summary_train < len(summary_train[i].split(' ')):
            max_length_summary_train = len(summary_train[i].split(' '))
    print("=== MAX SEQUENCE LENGTH SUMMARY TRAIN ===")
    print(max_length_summary_train)

    max_length_document_train = 0
    for i in range(len(document_train)):
        if max_length_document_train < len(document_train[i].split(' ')):
            max_length_document_train = len(document_train[i].split(' '))
    print("=== MAX SEQUENCE LENGTH DOCUMENT TRAIN ===")
    print(max_length_document_train)

    max_length_summary_test = 0
    for i in range(len(summary_test)):
        if max_length_summary_test < len(summary_test[i].split(' ')):
            max_length_summary_test = len(summary_test[i].split(' '))
    print("=== MAX SEQUENCE LENGTH SUMMARY TEST ===")
    print(max_length_summary_test)

    max_length_document_test = 0
    for i in range(len(document_test)):
        if max_length_document_test < len(document_test[i].split(' ')):
            max_length_document_test = len(document_test[i].split(' '))
    print("=== MAX SEQUENCE LENGTH DOCUMENT TEST ===")
    print(max_length_document_test)

    # printing the descriptive statistics of train and test datasets
    summary_train_lengths = pd.Series([len(x) for x in summary_train])
    print("=== DESCRIPTIVE STATISTICS SUMMARY TRAIN ===")
    print(summary_train_lengths.describe())

    document_train_lengths = pd.Series([len(x) for x in document_train])
    print("=== DESCRIPTIVE STATISTICS DOCUMENT TRAIN ===")
    print(document_train_lengths.describe())

    summary_test_lengths = pd.Series([len(x) for x in summary_test])
    print("=== DESCRIPTIVE STATISTICS SUMMARY TEST ===")
    print(summary_test_lengths.describe())

    document_test_lengths = pd.Series([len(x) for x in document_test])
    print("=== DESCRIPTIVE STATISTICS DOCUMENT TEST ===")
    print(document_test_lengths.describe())
