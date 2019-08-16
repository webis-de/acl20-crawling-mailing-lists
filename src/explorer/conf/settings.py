# Override settings in local_settings.py

# Flask secret
SECRET_KEY = None

# Elasticsearch settings
ES_SEED_HOSTS = ['localhost']
ES_INDEX = 'messages'

# Model paths
FASTTEXT_MODEL = 'email-vectors.bin'
SEGMENTER_MODEL = 'segmenter.hdf5'
