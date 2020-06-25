# Email Message Processing and Analysis

Research code for processing and analysing email and newsgroup messages.

The Webis-Gmane-19 email corpus was published at ACL 2020:

    @InProceedings{stein:2020o,
      author =              {Janek Bevendorff and Khalid Al-Khatib and Martin Potthast and Benno Stein},
      booktitle =           {58th Annual Meeting of the Association for Computational Linguistics (ACL 2020)},
      month =               jul,
      publisher =           {Association for Computational Linguistics},
      site =                {Seattle, USA},
      title =               {{Crawling and Preprocessing Mailing Lists At Scale for Dialog Analysis}},
      year =                2020
    }

The corpus itself can be found on [Zenodo](https://doi.org/10.5281/zenodo.3766984).

## Quickstart
Install dependencies via:

    pip3 install -r requirements.txt
    
The `run.sh` script can be used to start any of the tools and services
from the `src` directory with the correct `PYTHONPATH`.

### Train and Evaluate Model
Train model:

    ./run.sh src/parsing/message_segmenter.py train fasttext-model.bin \
        annotations/annotations-final-train.jsonl out/segmentation-model

Evaluate model:

    ./run.sh src/parsing/message_segmenter.py evaluate \
        trained-model.h5 fasttext-model.bin annotations/annotations-final-validation.jsonl

Pre-trained Fasttext and Tensorflow models can be found at [files.webis.de](https://files.webis.de/webis-gmane19-model/)

### Corpus Explorer
A web UI for data exploration can be found in `src/explorer/explorer.py`.
Before starting it, copy the main config file `src/explorer/conf/settings.py` to
`src/explorer/conf/local_settings.py` and adjust the config values (e.g. set the
correct model paths etc.)

The corpus explorer can be started using the `run.sh` script as follows:

    ./run.sh explorer [flask-options]

Note: the corpus explorer assumes you have indexed the Webis-Gmane-19 corpus to Elasticsearch.

### Other Tools in `src`
All command line tools in `src` can be started as follows:

    ./run.sh FILENAME

For individual usage instructions, run

    ./run.sh FILENAME --help
    
The following tools are available:

- `index/`
    - `corpus_extractpr.py`: Extractor for assembling final corpus
    - `mail_sampler.py`: Sample emails from Elasticsearch index
    - `message_index_annotator.py`: Segment and annotate message in an existing Elasticsearch index
    - `warc_indexer.py`: Index email WARC into Elasticsearch
- `parsing/`:
    - `message_segmenter.py`: Email message segmentation model (training, inference, evaluation)
    - `message_segmenter_svm.py`: Legacy email message segmentation model based on Tang et al., 2005
- `util/`:
    - Various other tools and libraries (see `--help` listings and doc strings)
