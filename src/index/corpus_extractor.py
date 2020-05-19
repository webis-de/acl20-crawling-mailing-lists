#!/usr/bin/env python3

from functools import partial
import gzip
import json
import os

import click

from util import util
from index.message_index_annotator import ANNOTATION_VERSION


logger = util.get_logger(__name__)


# noinspection PyIncorrectDocstring
@click.command()
@click.argument('index')
@click.argument('output_directory')
@click.option('-s', '--scroll-slices', help='Number of Elasticsearch scroll slices.', type=int, default=400)
@click.option('-x', '--scroll-size', help='Scroll size', type=int, default=400)
@click.option('-p', '--partitions', help='Number of output partitions (must be <= --scroll-slices).',
              type=int, default=250)
def main(index, output_directory, scroll_slices, scroll_size, partitions):
    """
    Extract final corpus with anonymised email addresses and updated annotations.

    Arguments:
        index: Elasticsearch index to extract messages from
        output_directory: output directory (shared folder on the cluster)
    """
    if partitions > scroll_slices:
        raise click.UsageError('--partitions must be less or equal to --scroll-slices.')

    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)

    sc = util.get_spark_context('Gmane Corpus Extractor', additional_conf={'spark.default.parallelism': scroll_slices})
    messages = sc.range(scroll_slices)
    messages = messages.repartition(scroll_slices)
    messages = messages.flatMap(partial(_retrieve_messages, max_slices=scroll_slices,
                                        scroll_size=scroll_size, index=index))
    messages = messages.coalesce(partitions)
    messages = messages.mapPartitionsWithIndex(partial(_write_to_gzip_files, output_dir=output_directory))
    messages.count()


def _retrieve_messages(slice_id, max_slices, scroll_size, index):
    logger.info('Retrieving initial batch (slice {}/{})'.format(slice_id, max_slices))
    es = util.get_es_client()
    results = util.es_retry(
        es.search, index=index, scroll='3h', request_timeout=360, size=scroll_size, body={
            "query": {
                "bool": {
                    "must": [
                        {"range": {"annotation_version": {"gte": ANNOTATION_VERSION}}},
                        {"wildcard": {"groupname": "gmane.*"}}
                    ]
                }
            },
            "sort": ["groupname", "@timestamp"],
            "_source": ["groupname", "@timestamp", "lang", "headers", "text_plain", "segments"],
            "slice": {
                "id": slice_id,
                "max": max_slices,
                "field": "id_hash"
            }
        })

    try:
        while results['hits']['hits']:
            batch = results['hits']['hits']

            for doc in batch:
                out_doc = doc['_source'].copy()

                # Rename, and filter fields
                out_doc['headers'] = {k: v for k, v in out_doc['headers'].items()if v and k in (
                    'message_id', 'from', 'to', 'cc', 'in_reply_to', 'references', 'subject', 'list_id'
                )}
                out_doc['headers']['date'] = out_doc.pop('@timestamp')
                out_doc['group'] = out_doc.pop('groupname')

                yield doc['_id'], out_doc

            logger.info('Retrieving next batch (slice {}/{})'.format(slice_id, max_slices))
            results = util.es_retry(es.scroll, scroll_id=results['_scroll_id'], scroll='3h', request_timeout=360)

    finally:
        es.clear_scroll(scroll_id=results['_scroll_id'])


def _write_to_gzip_files(part_id, batch, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    out_filename = os.path.join(output_dir, f'part-{part_id:04d}.ndjson.gz')

    f = None
    try:
        for i, (doc_id, message) in enumerate(batch):
            if i == 0:
                # Do not create file before starting to write anything to it
                f = gzip.open(out_filename, 'wb', compresslevel=9)
            elif i % 1000 == 0:
                # Start new gzip member segment every 1k lines
                f.close()
                f = gzip.open(out_filename, 'ab', compresslevel=9)

            action = {'index': {'_id': doc_id}}
            f.write('\n'.join((json.dumps(action), json.dumps(message), '')).encode())
    finally:
        if f is not None:
            f.close()

    return []


if __name__ == '__main__':
    main()
