import pandas as pd
import numpy as np


if __name__ == "__main__":
    """
    Main function for the data exploration of the train set
    """

    # read train and test set
    train_news = pd.read_csv("../dataset/train.csv")
    test_news = pd.read_csv("../dataset/train.csv")
    # dropping the irrelevant columns from the dataframe
    train_news.drop(['Source ', 'Time ', 'Publish Date'], axis=1, inplace=True)
    test_news.drop(['Source ', 'Time ', 'Publish Date'], axis=1, inplace=True)
    # printing the head of the dataset
    print("=== TRAIN DATASET HEAD ===")
    print(train_news.head())
    print("=== TRAIN DATASET HEAD ===")
    print(test_news.head())
    # printing the shape of the dataset
    print("=== DATASET SHAPE ===")
    print(news.shape)
    # printing the descriptive statistics of input and output column
    document, summary = news['Short'], news['Headline']
    document_lengths = pd.Series([len(x) for x in document])
    summary_lengths = pd.Series([len(x) for x in summary])
    print("=== DESCRIPTIVE STATISTICS DOCUMENT ===")
    print(document_lengths.describe())
    print("=== DESCRIPTIVE STATISTICS SUMMARY ===")
    print(summary_lengths.describe())
