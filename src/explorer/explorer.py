# Dataset explorer web application.

from elasticsearch import Elasticsearch
from keras import models
from flask import Flask, jsonify, render_template, request
from parsing.message_segmenter import predict_raw_text, load_fasttext_model

app = Flask(__name__)
app.config.from_object('conf.settings')
app.config.from_object('conf.local_settings')

es = Elasticsearch(app.config.get('ES_SEED_HOSTS'), sniff_on_start=True,
                   sniff_on_connection_fail=True, timeout=140)
load_fasttext_model(app.config.get('FASTTEXT_MODEL'))
line_model = models.load_model(app.config.get('SEGMENTER_MODEL'))
line_model._make_predict_function()


@app.route('/')
def index_route():
    return render_template('viewer.html', page_tpl='index.html')


@app.route('/query-mails', methods=['POST'])
def query_mails():
    query = request.get_json()
    return jsonify(es.search(index=app.config.get('ES_INDEX'), body=query).get('hits'))


@app.route('/predict-lines', methods=['POST'])
def predict_lines():
    predictions = list(predict_raw_text(line_model, request.data.decode('utf-8')))
    return jsonify(predictions)


if __name__ == '__main__':
    app.run(port=5001)
