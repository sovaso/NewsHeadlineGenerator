from main_class import NewsHeadlineGenerator


if __name__ == "__main__":
    """
    The main function to run the training
    """

    news_headline_generator = NewsHeadlineGenerator(number_of_layers=4,
                                                    d_model=128,
                                                    dff=512,
                                                    number_of_heads=8,
                                                    epochs=20,
                                                    encoder_max_length=400,
                                                    decoder_max_length=75,
                                                    buffer_size=20000,
                                                    batch_size=64)
    news_headline_generator.import_dataset("../dataset/inshorts.xlsx", document_column_name='Short',
                                           summary_column_name='Headline')
    news_headline_generator.preprocess_data()
    news_headline_generator.train_transformer()
