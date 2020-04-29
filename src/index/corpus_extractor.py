#!/usr/bin/env python3

from base64 import b64encode
from functools import partial
import gzip
from hashlib import sha256
import json
import os
import re

import click
from tensorflow.keras import models

from parsing import message_segmenter
from util import mail_classification, util


logger = util.get_logger(__name__)


@click.command()
@click.argument('index')
@click.argument('output_directory')
@click.argument('segmentation_model', type=click.Path(exists=True, dir_okay=False))
@click.argument('fasttext_model', type=click.Path(exists=True, dir_okay=False))
@click.option('-s', '--slices', help='Number of Elasticsearch scroll slices', type=int, default=200)
@click.option('-x', '--scroll-size', help='Scroll size (should be quite small to avoid scroll context timeouts)',
              type=int, default=50)
@click.option('-p', '--output-file-prefix', help='Export index name', default='webis_gmane_corpus_2019')
def main(index, output_directory, segmentation_model, fasttext_model, **kwargs):
    """
    Extract final corpus with anonymised email addresses and updated annotations.

    Arguments:
        index: Elasticsearch index to sample from
        output_directory: output directory (shared folder on the cluster)
        segmentation_model: pre-trained HDF5 email segmentation model
        fasttext_model: pre-trained FastText embedding
    """
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)

    sc = util.get_spark_context('Gmane Corpus Extractor')
    sc.range(0, kwargs['slices']).foreach(partial(
        _start_spark_worker, index=index, output_directory=output_directory, segmentation_model=segmentation_model,
        fasttext_model=fasttext_model, max_slices=kwargs['slices'], **kwargs))


def _start_spark_worker(slice_id, index, output_directory, segmentation_model, fasttext_model, max_slices=1, **kwargs):
    logger.info('Loading segmentation model')
    mail_classification.load_fasttext_model(fasttext_model)
    segmentation_model = models.load_model(segmentation_model)

    logger.info('Retrieving initial batch (slice {}/{})'.format(slice_id, max_slices))
    es = util.get_es_client()
    results = util.es_retry(
        es.search, index=index, scroll='1h', request_timeout=360, size=kwargs.get('scroll_size', 2000), body={
            "query": {
                "wildcard": {"groupname": "gmane.*"}
            },
            "_source": ["groupname", "@timestamp", "lang", "headers", "text_plain"],
            "slice": {
                "id": slice_id,
                "max": max_slices,
                "field": "@timestamp"
            }
        })

    email_regex = re.compile(r'((?:[a-zA-Z0-9_\-./+]+)@(?:(?:\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.)|' +
                             r'(?:(?:[a-zA-Z0-9\-]+\.)+))(?:[a-zA-Z]{2,}|[0-9]{1,3})(?:\]?))')

    def email_replacement(e):
        return b64encode(sha256(e.group().encode()).digest())[:16].decode() + '@example.com'

    def extract_email_addr(s):
        try:
            return email_regex.search(s).group()
        except AttributeError:
            # Use value as is if not an email address
            return s

    output_file_prefix = kwargs.get('output_file_prefix', 'webis_gmane_corpus_2019')
    output_buffer = b''

    def flush_out_file():
        nonlocal output_buffer
        out_file.write(gzip.compress(output_buffer, compresslevel=9))
        output_buffer = b''
        out_file.flush()

    try:
        out_file_name = os.path.join(output_directory, '{}_part{:03d}.ndjson.gz'.format(output_file_prefix, slice_id))
        with open(out_file_name, 'wb') as out_file:
            while results['hits']['hits']:
                batch = results['hits']['hits']

                for doc in batch:
                    src = doc['_source']
                    # Anonymise email addresses
                    src['body'] = email_regex.sub(email_replacement, src.pop('text_plain'))
                    src['headers'] = {k: src['headers'][k] for k in src['headers']
                                      if src['headers'][k] and k in
                                      {'message_id', 'from', 'to', 'cc', 'in_reply_to', 'references', 'list_id'}}

                    for h in src['headers']:
                        if h in {'cc', 'references'}:
                            if type(src['headers'][h]) is not list:
                                src['headers'][h] = [src['headers'][h]]
                                src['headers'][h] = [extract_email_addr(x) for x in src['headers'][h]]

                        # Keep only email address in from
                        if h == 'from':
                            src['headers'][h] = extract_email_addr(src['headers'][h])

                        if type(src['headers'][h]) is list:
                            src['headers'][h] = [email_regex.sub(email_replacement, x) for x in src['headers'][h]]
                        elif h != 'list_id':
                            src['headers'][h] = email_regex.sub(email_replacement, src['headers'][h])

                    # Rename fields
                    src['headers']['date'] = src.pop('@timestamp')
                    src['group'] = src.pop('groupname')

                    src['segments'] = []
                    try:
                        lines = message_segmenter.predict_raw_text(segmentation_model, src['body'])
                        src['segments'] = [{'start': s[0], 'end': s[1], 'label': s[2]}
                                           for s in mail_classification.line_labels_to_char_pos(lines)]
                    except Exception as e:
                        logger.error('Error segmenting message: {}'.format(e))

                    index_action = {'index': {'_id': doc['_id']}}
                    output_buffer += (json.dumps(index_action) + '\n' + json.dumps(src) + '\n').encode()

                    # Flush if larger than 10MiB
                    if len(output_buffer) > 10 * 1024 * 1024:
                        flush_out_file()

                logger.info('Retrieving next batch (slice {}/{})'.format(slice_id, max_slices))
                results = util.es_retry(es.scroll, scroll_id=results['_scroll_id'], scroll='1h', request_timeout=360)
    finally:
        flush_out_file()
        out_file.close()
        es.clear_scroll(scroll_id=results['_scroll_id'])


if __name__ == '__main__':
    main()
