#!/usr/bin/env python3

from base64 import b64encode
from collections import defaultdict
import itertools
from functools import partial
from hashlib import sha256
import os
import re
import site
import sys
from time import time

import click
from elasticsearch import helpers
import spacy
from spacy_langdetect import LanguageDetector
from tensorflow.keras import models
from tqdm import tqdm

from parsing.message_segmenter import load_fasttext_model, predict_raw_text
from util import mail_classification, util


ANNOTATION_VERSION = 11

logger = util.get_logger(__name__)


@click.command()
@click.argument('index')
@click.argument('segmentation_model', type=click.Path(exists=True, dir_okay=False))
@click.argument('fasttext_model', type=click.Path(exists=True, dir_okay=False))
@click.option('-s', '--scroll-slices', help='Number of Elasticsearch scroll slices', type=int, default=200)
@click.option('-x', '--scroll-size', help='Scroll size', type=int, default=150)
@click.option('-n', '--dry-run', help='Dry run (do not index anything)', is_flag=True)
@click.option('-a', '--anonymize', help='Anonymize email addresses', is_flag=True)
def main(index, segmentation_model, fasttext_model, **kwargs):
    """
    Automatic message index annotation tool.

    Automatically annotates an existing message index by scrolling through all messages,
    analyzing them, and updating the index documents with the generated annotations.

    Arguments:
        index: the Elasticsearch index
        segmentation_model: pre-trained HDF5 email segmentation model
        fasttext_model: pre-trained FastText embedding
    """

    start_indexer(index, segmentation_model, fasttext_model, **kwargs)


def start_indexer(index, segmentation_model, fasttext_model, **kwargs):
    """
    Start annotation indexer.

    :param index: Elasticsearch index
    :param segmentation_model: HDF5 Email Segmentation model
    :param fasttext_model: fastText email embedding

    Keyword Args:
        dry_run (bool): Perform dry run, do not actually index anything
        progress_bar (bool): Show indexing progress bar
    """

    if kwargs.get('dry_run'):
        logger.warning('Started in dry run mode, nothing will be indexed.')

    es = util.get_es_client()

    if not es.indices.exists(index=index):
        raise RuntimeError('Index has to exist.')

    logger.info('Updating Elasticsearch index mapping')
    es.indices.put_mapping(index=index, body={
        "properties": {
            "main_content": {
                "type": "text"
            },
            "segments": {
                "type": "nested",
                "properties": {
                    "begin": {
                        "type": "integer"
                    },
                    "end": {
                        "type": "integer"
                    },
                    "label": {
                        "type": "keyword"
                    },
                }
            },
            "label_stats": {
                "properties": {
                    "paragraph_quotation.num_ratio": {"type": "float"},
                    "paragraph_quotation.lines_ratio": {"type": "float"}
                }
            },
            "annotation_version": {
                "type": "short"
            }
        },
        "dynamic": True,
        "dynamic_templates": [
            {
                "stats": {
                    "path_match": "label_stats.*",
                    "mapping": {
                        "properties": {
                            "num": {"type": "integer"},
                            "chars": {"type": "integer"},
                            "lines": {"type": "integer"},
                            "avg_len": {"type": "float"}
                        }
                    }
                }
            }
        ]
    })

    slices = kwargs.get('scroll_slices', 2)
    sc = util.get_spark_context('Mail Annotation Indexer', additional_conf={'spark.default.parallelism': slices})
    rdd = sc.range(0, slices)
    rdd = rdd.repartition(slices)
    rdd.foreach(partial(_start_spark_worker, index=index, segmentation_model=segmentation_model,
                        fasttext_model=fasttext_model, **kwargs))


def _start_spark_worker(slice_id, index, segmentation_model, fasttext_model, **kwargs):
    # Fix to circumvent Yarn's buggy HOME override
    os.environ['HOME'] = os.environ.get('HADOOP_HOME', os.environ['HOME'])

    logger.info('Loading SpaCy')
    if not spacy.util.is_package('en_core_web_sm'):
        oldbase = site.USER_BASE
        site.USER_BASE = os.path.join(os.environ['HOME'], '.local')
        site.USER_SITE = site.USER_SITE.replace(oldbase, site.USER_BASE)
        os.makedirs(site.USER_SITE, exist_ok=True)
        sys.path.insert(0, site.USER_SITE)
        spacy.cli.download('en_core_web_sm')
    import en_core_web_sm
    nlp = en_core_web_sm.load()
    nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)

    logger.info('Loading segmentation model')
    load_fasttext_model(fasttext_model)
    segmentation_model = models.load_model(segmentation_model)

    max_slices = kwargs.get('scroll_slices', 2)
    logger.info('Retrieving initial batch (slice {}/{})'.format(slice_id, max_slices))
    es = util.get_es_client()
    results = util.es_retry(es.search, index=index, scroll='45m', size=kwargs['scroll_size'], body={
        'sort': ['_id'],
        'slice': {
            'id': slice_id,
            'max': max_slices,
            'field': 'id_hash'
        },
        'query': {
            'bool': {
                "must": {
                    "wildcard": {"group": "gmane.*"}
                },
                'must_not': {
                    'range': {'annotation_version': {'gte': ANNOTATION_VERSION}}
                }
            }
        }
    })

    try:
        while results['hits']['hits']:
            logger.info('Processing batch.')
            doc_gen = _generate_docs(results['hits']['hits'], index, segmentation_model, nlp,
                                     progress_bar=False, anonymize=kwargs.get('anonymize', False))
            try:
                if kwargs.get('dry_run'):
                    while True:
                        next(doc_gen)
                else:
                    # only start bulk request if generator has at least one element
                    peek = next(doc_gen)
                    helpers.bulk(es, itertools.chain([peek], doc_gen))
                logger.info('Finished indexing batch.')
            except StopIteration:
                pass

            logger.info('Retrieving next batch (slice {}/{})'.format(slice_id, max_slices))
            results = util.es_retry(es.scroll, scroll_id=results['_scroll_id'], scroll='45m')
    finally:
        es.clear_scroll(scroll_id=results['_scroll_id'])


def _generate_docs(batch, index, segmentation_model, nlp, progress_bar=False, anonymize=False):
    """
    Generate Elasticsearch index docs.

    :param batch: batch of documents
    :param index: Elasticsearch index
    :param segmentation_model: Email segmentation model
    :param nlp: SpaCy language model
    :return: Generator of index doc actions

    Keyword Args:
        See :func:`_start_spark_worker`
    """

    if progress_bar:
        batch = tqdm(batch, desc='Preparing documents in batch', unit='docs', total=len(batch), leave=False)

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

    for doc in batch:
        doc_id = doc['_id']

        if not doc_id:
            continue

        if doc.get('annotation_version', -1) >= ANNOTATION_VERSION:
            logger.error('{}: document annotation version greater or equal {}.'.format(doc_id, ANNOTATION_VERSION))
            continue

        src = doc['_source']

        # Remove invalid surrogate characters
        raw_text = src.get('text_plain', '').encode('utf-8', 'ignore').decode('utf-8')

        output_doc = {}

        # Anonymize email addresses
        if anonymize:
            raw_text = email_regex.sub(email_replacement, raw_text)
            output_doc['text_plain'] = raw_text

            src['headers'] = {k: src['headers'][k] for k in src['headers']
                              if src['headers'][k] and k in {'message_id', 'subject', 'from', 'to', 'cc',
                                                             'in_reply_to', 'references', 'list_id'}}
            for h in src['headers']:
                # Keep only email address in certain headers
                if h in {'cc', 'to'}:
                    if type(src['headers'][h]) is not list:
                        src['headers'][h] = [src['headers'][h]]
                    src['headers'][h] = [extract_email_addr(x) for x in src['headers'][h]]
                elif h in {'from', 'from_email', 'in_reply_to'}:
                    src['headers'][h] = extract_email_addr(src['headers'][h])

                if type(src['headers'][h]) is list:
                    src['headers'][h] = [email_regex.sub(email_replacement, x) for x in src['headers'][h]]
                elif h != 'list_id':
                    src['headers'][h] = email_regex.sub(email_replacement, src['headers'][h])

            output_doc['headers'] = src['headers']

        stats = defaultdict(lambda: {'num': 0, 'chars': 0, 'lines': 0, 'avg_len': 0.0})
        occurrences = defaultdict(lambda: 0)

        # Segment message, but  skip overly long texts to avoid running out of memory
        if len(raw_text) <= 100000:
            try:
                logger.debug('Segmenting message')
                label_gen = predict_raw_text(segmentation_model, raw_text)
            except Exception as e:
                logger.error('Error segmenting message: {}'.format(e))
                label_gen = []

            # Calculate segment stats and extract main content
            logger.debug('Calculating segment stats')
            main_content = ''
            output_doc['segments'] = []
            for begin, end, label in mail_classification.line_labels_to_char_pos(label_gen):
                if label in ['paragraph', 'section_heading']:
                    main_content += raw_text[begin:end]

                stats[label]['num'] += 1
                stats[label]['chars'] += end - begin
                stats[label]['lines'] += raw_text[begin:end].count('\n')
                occurrences[label] += 1

                output_doc['segments'].append({'begin': begin, 'end': end, 'label': label})

            # Collapse newlines to a maximum of two
            main_content = re.sub(r'\n{3,}', '\n\n', main_content).rstrip()

            output_doc['main_content'] = main_content

            for label in stats:
                stats[label]['avg_len'] = stats[label]['chars'] / occurrences[label]

            stats['paragraph_quotation'] = {
                'num_ratio': (stats['paragraph']['num'] / stats['quotation']['num'])
                if stats['quotation']['num'] > 0 else -1,

                'lines_ratio': (stats['paragraph']['lines'] / stats['quotation']['lines'])
                if stats['quotation']['lines'] > 0 else -1,
            }

            output_doc['label_stats'] = dict(stats)

            # Improve language prediction by making use of content segmentation
            if len(main_content) > 15:
                logger.debug('Detecting language')
                output_doc['lang'] = nlp(main_content)._.language['language']
        else:
            logger.warning('Skipped overly long message ({} bytes).'.format(len(raw_text)))

        output_doc['annotation_version'] = ANNOTATION_VERSION
        output_doc['modified'] = int(time() * 1000)

        yield {
            '_op_type': 'update',
            '_index': index,
            '_type': 'message',
            '_id': doc_id,
            'doc': output_doc
        }


if __name__ == '__main__':
    main()
