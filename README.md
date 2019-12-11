# Email Message Processing and Analysis

Research code for processing and analysing email and newsgroup messages.

Install dependencies via:

    pip3 install -r requirements.txt
    
The `run.sh` script can be used to start any of the tools and services
from the `src` directory with the correct `PYTHONPATH`.


## Corpus Explorer
A web UI for data exploration can be found in `src/explorer/explorer.py`.
Before starting it, copy the main config file `src/explorer/conf/settings.py` to
`src/explorer/conf/local_settings.py` and adjust the config values (e.g. set the
correct model paths etc.)

The corpus explorer can be started using the `run.sh` script as follows:

    ./run.sh explorer [flask-options]

## Other Tools in `src`
- `'index/`
    - `mail_sampler.py`: Sample emails from Elasticsearch index
    - `message_index_annotator.py`: Segment and annotate message in an existing Elasticsearch index
    - `warc_indexer.py`: Index email WARC into Elasticsearch
- `parsing/`:
    - `message_segmenter.py`: Email message segmentation model (training, inference, evaluation)
    - `message_segmenter_svm.py`: Legacy email message segmentation model based on Tang et al., 2005
- `util/`: Various tools and libraries

