import pandas as pd


if __name__ == "__main__":
    """
    Splitting the original dataset on train and test set with the ratio of 99% and 1% of original dataset
    """

    # reading the dataset
    news = pd.read_excel("../dataset/inshorts.xlsx")
    # creating train dataframe
    news_train_df = news.sample(frac=0.99, random_state=2211)
    # creating test dataframe
    news_test_df = news.drop(news_train_df.index)
    # creating train and test dataset as csv files
    news_train_df.to_csv("../dataset/train.csv")
    news_test_df.to_csv("../dataset/test.csv")
