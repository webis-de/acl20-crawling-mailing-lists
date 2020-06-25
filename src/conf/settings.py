# Override settings in local_settings.py

# Flask secret
SECRET_KEY = None

# Elasticsearch settings
ES_SEED_HOSTS = [{'host': 'localhost', 'port': 9200}]
ES_CONNECTION_PROPERTIES = {
    'use_ssl': True,
    'api_key': ('id', 'api_key'),
    'sniff_on_start': True,
    'sniff_on_connection_fail': True
}
ES_INDEX = 'webis_gmane_19'

# Explorer Web UI Model paths
FASTTEXT_MODEL = 'fasttext-model.bin'
SEGMENTER_MODEL = 'segmenter.h5'
