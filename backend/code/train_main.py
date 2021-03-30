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
    #outputtt = news_headline_generator.test_transformer("After veteran Congress leader Rita Bahuguna Joshi joined the BJP on Thursday, she said in a press conference that Rahul Gandhi was unable to provide the kind of leadership that a national party like Congress needs. She further said that Congress President Sonia Gandhi used to listen to party members but that was not possible under Rahul Gandhi&#39;s leadership.")
    #print("OUTPUT ->", outputtt)
