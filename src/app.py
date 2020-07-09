from flask import Flask, request
from flask_restful import Api, Resource
import torch
from src import bert_config,masked_language_modeling,gpt2_config,generate_text

app = Flask(__name__)
api = Api(app)

@app.route('/maskedlm', methods=['POST'])
def getMaskedLMResult():
    request_data = request.get_json()
    textData = request_data['articleDetails']

    comp_text=masked_language_modeling.completed_outcome_maskedLM(textData)

    return {'completed_text': comp_text}


@app.route('/generate', methods=['POST'])
def generateArticle():
    request_data = request.get_json()
    print(request_data)
    textData = request_data['articleDetails']
    no_of_samples=request_data['noOfSamples']
    word_length=request_data['length']
    temperature=request_data['temperature']

    generated_text_list=generate_text.generate_article(textData,no_of_samples,
                                                       word_length,temperature)

    return {'generated_text': generated_text_list}


if __name__ == '__main__':
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    print(device)
    bert_config.BERT_MODEL.eval()
    gpt2_config.GPT2_MODEL.eval()
    app.run()

