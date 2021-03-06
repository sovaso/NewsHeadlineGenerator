import pandas as pd
import numpy as np


if __name__ == "__main__":
    """
    Main function for the data exploration of the train set
    """

    # reading the dataset
    news = pd.read_excel("../dataset/inshorts.xlsx")
    # dropping the irrelevant columns from the dataframe
    news.drop(['Source ', 'Time ', 'Publish Date'], axis=1, inplace=True)
    # printing the head of the dataset
    print("=== DATASET HEAD ===")
    print(news.head())
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
