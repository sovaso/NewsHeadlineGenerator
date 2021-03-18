from code.main_class import NewsHeadlineGenerator
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/predict', methods=['POST'])
def predict():
    print(request.data.decode("utf-8"))
    input = request.data.decode("utf-8")
    news_headline_generator = NewsHeadlineGenerator(number_of_layers=4, d_model=128, dff=512, number_of_heads=8,
                                                    epochs=20, encoder_max_length=400, decoder_max_length=75,
                                                    buffer_size=20000, batch_size=64)
    news_headline_generator.import_dataset("../dataset/inshorts.xlsx", document_column_name='Short',
                                           summary_column_name='Headline')
    news_headline_generator.preprocess_data()
    """
    output = news_headline_generator.test_transformer("The CBI on Saturday booked four former officials of Syndicate Bank and six others for cheating, forgery, criminal conspiracy and causing ₹209 crore loss to the state-run bank. The accused had availed home loans and credit from Syndicate Bank on the basis of forged and fabricated documents. These funds were fraudulently transferred to the companies owned by the accused persons.")
    """
    output = news_headline_generator.test_transformer(input)
    print('Output -> ', output)
    return jsonify(output)


if __name__ == '__main__':
    app.run(debug=True)
