#!/usr/bin/env python3

from base64 import b64encode
from functools import partial
import gc
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
@click.option('-s', '--scroll-slices', help='Number of Elasticsearch scroll slices', type=int, default=100)
@click.option('-x', '--scroll-size', help='Scroll size', type=int, default=100)
@click.option('-o', '--output-partitions', help='Number of output partitions', type=int, default=200)
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
    messages = sc.range(kwargs['scroll_slices']).flatMap(partial(_map_messages, max_slices=kwargs['scroll_slices'],
                                                                 scroll_size=kwargs['scroll_size'], index=index))
    num_msg = util.get_es_client().count(index, body={"query": {"wildcard": {"groupname": "gmane.*"}}})['count']
    messages.partitionBy(min(1, num_msg // 1000))
    messages = messages.mapPartitions(partial(_predict_segments, segmentation_model=segmentation_model,
                                              fasttext_model=fasttext_model),
                                      preservesPartitioning=True)
    messages.partitionBy(min(1, num_msg // 10000))
    messages = messages.mapPartitionsWithIndex(partial(_create_output_segments, output_base_dir=output_directory,
                                                       num_output_partitions=kwargs['output_partitions']))

    # Force execution
    messages.count()

    # We don't need this anymore
    del messages
    gc.collect()

    sc.range(kwargs['output_partitions']).foreach(partial(_combine_output_segments, output_base_dir=output_directory,
                                                          output_file_prefix=kwargs['output_file_prefix']))


def _map_messages(slice_id, max_slices, scroll_size, index):
    logger.info('Retrieving initial batch (slice {}/{})'.format(slice_id, max_slices))
    es = util.get_es_client()
    results = util.es_retry(
        es.search, index=index, scroll='30m', request_timeout=360, size=scroll_size, body={
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
        except (TypeError, AttributeError):
            # Use value as is if not an email address
            logger.warning('Expected email address in "{}", but couldn\'t find any.'.format(s))
            return s

    try:
        while results['hits']['hits']:
            batch = results['hits']['hits']

            for doc in batch:
                src = doc['_source']
                # Anonymise email addresses
                src['body'] = email_regex.sub(email_replacement, src.pop('text_plain'))
                src['headers'] = {k: src['headers'][k] for k in src['headers']
                                  if src['headers'][k] and k in {'message_id', 'subject', 'from', 'to', 'cc',
                                                                 'in_reply_to',  'references', 'list_id'}}

                for h in src['headers']:
                    # Keep only email address in certain headers
                    if h in {'cc', 'to'}:
                        if type(src['headers'][h]) is not list:
                            src['headers'][h] = [src['headers'][h]]
                        src['headers'][h] = [extract_email_addr(x) for x in src['headers'][h]]
                    elif h in {'from', 'in_reply_to'}:
                        src['headers'][h] = extract_email_addr(src['headers'][h])

                    if type(src['headers'][h]) is list:
                        src['headers'][h] = [email_regex.sub(email_replacement, x) for x in src['headers'][h]]
                    elif h != 'list_id':
                        src['headers'][h] = email_regex.sub(email_replacement, src['headers'][h])

                # Rename fields
                src['headers']['date'] = src.pop('@timestamp')
                src['group'] = src.pop('groupname')

                yield doc['_id'], src

            logger.info('Retrieving next batch (slice {}/{})'.format(slice_id, max_slices))
            results = util.es_retry(es.scroll, scroll_id=results['_scroll_id'], scroll='30m', request_timeout=360)

    finally:
        es.clear_scroll(scroll_id=results['_scroll_id'])


def _predict_segments(messages, segmentation_model, fasttext_model):
    logger.info('Loading segmentation model')
    mail_classification.load_fasttext_model(fasttext_model)
    segmentation_model = models.load_model(segmentation_model)

    logger.info('Predicting segments')
    for msg_id, msg in messages:
        msg['segments'] = []
        try:
            lines = message_segmenter.predict_raw_text(segmentation_model, msg['body'])
            msg['segments'] = [{'start': s[0], 'end': s[1], 'label': s[2]}
                               for s in mail_classification.line_labels_to_char_pos(lines)]
        except Exception as e:
            logger.error('Error segmenting message: {}'.format(e))

        index_action = {'index': {'_id': msg_id}}
        yield (json.dumps(index_action) + '\n' + json.dumps(msg) + '\n').encode()


def _create_output_segments(part_id, message_bytes_gen, output_base_dir, num_output_partitions):
    out_dir = os.path.join(output_base_dir, 'segment_{:05d}'.format(part_id % num_output_partitions))
    os.makedirs(out_dir, exist_ok=True)

    with gzip.open(os.path.join(out_dir, 'part_{:08d}.ndjson.gz'.format(part_id)), 'wb', compresslevel=9) as f:
        for message in message_bytes_gen:
            f.write(message)

    return []


def _combine_output_segments(output_part_id, output_base_dir, output_file_prefix):
    segment_dir = os.path.join(output_base_dir, 'segment_{:05d}'.format(output_part_id))
    out_file = os.path.join(output_base_dir, '{}_part{:04d}.ndjson.gz'.format(output_file_prefix, output_part_id))

    if not os.path.isdir(segment_dir):
        logger.error('Segment dir {} does not exist.'.format(segment_dir))
        return

    infiles = [os.path.join(segment_dir, i) for i in os.listdir(segment_dir) if i.endswith('.ndjson.gz')]
    with open(out_file, 'wb') as f:
        logger.info('Merging segments of part {}'.format(output_part_id))
        for infile in infiles:
            f.write(open(infile, 'rb').read())
            f.flush()

    for infile in infiles:
        os.remove(infile)

    if not os.listdir(segment_dir):
        os.rmdir(segment_dir)


if __name__ == '__main__':
    main()
