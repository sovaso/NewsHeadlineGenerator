from code.main_class import NewsHeadlineGenerator
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Service used to predict and generate the headline from sequence which is sent from client side (frontend)
    """

    print(request.data.decode("utf-8"))
    input = request.data.decode("utf-8")
    news_headline_generator = NewsHeadlineGenerator(number_of_layers=4,
                                                    d_model=128,
                                                    dff=512,
                                                    number_of_heads=8,
                                                    epochs=100,
                                                    batch_size=64,
                                                    encoder_max_length=89,
                                                    decoder_max_length=20)
    news_headline_generator.import_train_and_test_set("../dataset")
    output = news_headline_generator.test_transformer(input)
    print('OUTPUT -> ', output)
    return jsonify(output)


if __name__ == '__main__':
    """
    Simple Flask app to demonstrate the work of Transformer model via full stack application
    """

    app.run(debug=True)
