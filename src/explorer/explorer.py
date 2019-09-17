# Dataset explorer web application.

from elasticsearch import Elasticsearch
from tensorflow.python.keras import models
from flask import Flask, abort, jsonify, render_template, request
from parsing.message_segmenter import predict_raw_text, load_fasttext_model, reformat_raw_text_recursive
import util.util as util

app = Flask(__name__)
app.config.from_object('conf.settings')
app.config.from_object('conf.local_settings')

es = Elasticsearch(app.config.get('ES_SEED_HOSTS'), sniff_on_start=True,
                   sniff_on_connection_fail=True, timeout=140)
load_fasttext_model(app.config.get('FASTTEXT_MODEL'))
line_model = models.load_model(app.config.get('SEGMENTER_MODEL'))


@app.route('/')
def index_route():
    return render_template('index.html', page_tpl='index.html')


@app.route('/query-mails', methods=['POST'])
def query_mails():
    query = request.get_json()
    return jsonify(es.search(index=app.config.get('ES_INDEX'), body=query).get('hits'))


@app.route('/predict-lines', methods=['POST'])
def predict_lines():
    predictions = list(predict_raw_text(line_model, request.data.decode('utf-8')))
    return jsonify(predictions)


@app.route('/reformat-mail', methods=['POST'])
def reformat_mail():
    predictions = list(reformat_raw_text_recursive(line_model, request.data.decode('utf-8')))
    return jsonify(predictions)


@app.route('/get-thread', methods=['GET'])
def get_thread():
    if not request.args.get('message_id'):
        abort(400, 'Missing message_id')
    return jsonify(util.retrieve_email_thread(es, app.config.get('ES_INDEX'), request.args.get('message_id')))


if __name__ == '__main__':
    app.run(port=5001)
