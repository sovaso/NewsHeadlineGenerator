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
                                                    batch_size=64,
                                                    encoder_max_length=89,
                                                    decoder_max_length=20)
    news_headline_generator.import_train_and_test_set("../dataset")
    #news_headline_generator.train_transformer()
    outputtt = news_headline_generator.test_transformer("A hot air balloon, part of a Goa Tourism Initiative, caused panic among residents when it landed in Raia village on Saturday. The villagers ran for shelter and called the police. Earlier this week, Goa Tourism Minister Manohar Ajgaonkar ordered an inquiry after a hot air balloon carrying six tourists landed near a house in Shiroda village. ")
    print("OUTPUT ->", outputtt)
