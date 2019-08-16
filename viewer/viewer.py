from elasticsearch import Elasticsearch
from keras import models
from flask import Flask, jsonify, render_template, request
from mail_cleanup_deep import predict_raw_email
from threading import Lock

app = Flask(__name__)
app.config.from_object('local_settings')

es = Elasticsearch(['betaweb015', 'betaweb020'], sniff_on_start=True, sniff_on_connection_fail=True, timeout=140)
line_model = models.load_model('data/line_model_2k_hinge2.epoch-14.loss-0.21.hdf5')
line_model._make_predict_function()

lock = Lock()


@app.route('/')
def index_route():
    return render_template('viewer.html', page_tpl='index.html')


@app.route('/query-mails', methods=['POST'])
def query_mails():
    query = request.get_json()
    return jsonify(es.search(index='gmane_messages_all', body=query).get('hits', {}).get('hits'))


@app.route('/predict-lines', methods=['POST'])
def predict_lines():
    with lock:
        predictions = list(predict_raw_email(line_model, request.data.decode('utf-8')))
    return jsonify(predictions)


if __name__ == '__main__':
    app.run(port=5001)
