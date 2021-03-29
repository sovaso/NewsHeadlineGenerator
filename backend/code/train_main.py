from main_class import NewsHeadlineGenerator


if __name__ == "__main__":
    """
    The main function to run the training
    """

    news_headline_generator = NewsHeadlineGenerator(number_of_layers=4,
                                                    d_model=128,
                                                    dff=512,
                                                    number_of_heads=8,
                                                    epochs=100,
                                                    batch_size=64)
    news_headline_generator.import_train_and_test_set("../dataset")
    news_headline_generator.train_transformer()
