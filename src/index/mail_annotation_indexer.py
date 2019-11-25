#!/usr/bin/env python3
#
# Index Doccano JSON annotations to Elasticsearch.
# The index is only updated and must exist.

from parsing.message_segmenter import load_fasttext_model, predict_raw_text
from util.util import get_es_client, get_spark_context

import click
from elasticsearch import helpers
from collections import defaultdict
from functools import partial
from glob import glob
import itertools
import json
import logging
import os
import re
import spacy
from spacy_langdetect import LanguageDetector
from tensorflow.python.keras import models
from time import time
from tqdm import tqdm

ANNOTATION_VERSION = 4

logging.basicConfig(level=os.environ.get('LOGLEVEL', logging.INFO))
logger = logging.getLogger(__name__)


@click.command()
@click.argument('index')
@click.option('-m', '--model', help='Line segmentation model', type=click.Path(exists=True, dir_okay=False))
@click.option('-f', '--fmodel', help='Fasttext model', type=click.Path(exists=True, dir_okay=False))
@click.option('-i', '--input-file', help='Input annotation file', type=click.Path(exists=True, dir_okay=False))
@click.option('-s', '--slices', help='Index scroll slices', type=int, default=100)
@click.option('-a', '--arg-lexicon', help='Arguing Lexicon directory (Somasundaran et al., 2007)',
              type=click.Path(exists=True, file_okay=False))
def main(index, model, fmodel, input_file, slices, arg_lexicon):
    if model and not fmodel:
        raise click.UsageError('FastText model is required if segmentation model is specified.')

    if not model and not input_file:
        raise click.UsageError('Need to specify either segmentation model or input annotation file.')

    if arg_lexicon:
        logger.info('Loading Arguing Lexicon...')
        arg_lexicon = load_arglex(arg_lexicon)

    start_indexer(index, model, fmodel, input_file, slices, arg_lexicon)


def start_indexer(index, model, fasttext_model, input_file, slices, arg_lexicon):
    es = get_es_client()

    if not es.indices.exists(index=index):
        raise RuntimeError('Index has to exist.')

    logger.info('Updating Elasticsearch index mapping...')
    es.indices.put_mapping(index=index, doc_type='message', body={
                'properties': {
                    'label_stats': {
                        'properties': {
                            'paragraph_quotation.num_ratio': {'type': 'float'},
                            'paragraph_quotation.lines_ratio': {'type': 'float'}
                        }
                    },
                    'annotation_version': {
                        'type': 'short'
                    },
                    'arg_classes': {
                        'type': 'keyword'
                    },
                    'arg_classes_matched_regex': {
                        'type': 'keyword'
                    }
                },
                'dynamic': True,
                'dynamic_templates': [
                    {
                        'stats': {
                            'path_match': 'label_stats.*',
                            'mapping': {
                                'type': 'object',
                                'properties': {
                                    'num': {'type': 'integer'},
                                    'chars': {'type': 'integer'},
                                    'lines': {'type': 'integer'},
                                    'avg_len': {'type': 'float'}
                                }
                            }
                        }
                    }
                ]
    })

    if model and fasttext_model:
        sc = get_spark_context('Mail Annotation Indexer')
        sc.range(0, slices).foreach(partial(start_spark_worker,
                                            max_slices=slices,
                                            index=index,
                                            model=model,
                                            fasttext_model=fasttext_model,
                                            arg_lexicon=arg_lexicon))
    elif input_file:
        logger.info('Loading SpaCy...')
        nlp = spacy.load('en_core_web_sm')
        nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)

        logger.info('Indexing annotations...')
        helpers.bulk(es, generate_docs([json.loads(l) for l in open(input_file, 'r')], index, nlp=nlp))


def start_spark_worker(slice_id, max_slices, index, model, fasttext_model, arg_lexicon):
    logger.info('Loading SpaCy...')
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)

    logger.info('Loading segmentation model...')
    load_fasttext_model(fasttext_model)
    model = models.load_model(model)


    if arg_lexicon:
        logger.info('Compiling arguing lexicon regex list...')
        arg_lexicon = [(re.compile(regex, re.IGNORECASE), regex, cls) for regex, cls in arg_lexicon]

    logger.info('Retrieving initial batch (slice {}/{})...'.format(slice_id, max_slices))
    es = get_es_client()
    results = es.search(index=index, scroll='35m', size=250, body={
        'sort': ['_id'],
        'slice': {
            'id': slice_id,
            'max': max_slices,
            'field': '@timestamp'
        },
        'query': {
            'bool': {
                'must_not': {
                    'range': {'annotation_version': {'gte': ANNOTATION_VERSION}}
                }
            }
        }
    })

    while results['hits']['hits']:
        logger.info('Processing batch.')
        doc_gen = generate_docs(results['hits']['hits'], index=index, model=model,
                                nlp=nlp, arg_lexicon=arg_lexicon, progress_bar=False)
        try:
            # only start bulk request if generator has at least one element
            peek = next(doc_gen)
            helpers.bulk(es, itertools.chain([peek], doc_gen))
            logger.info('Finished indexing batch.')
        except StopIteration:
            pass

        logger.info('Retrieving next batch (slice {}/{})...'.format(slice_id, max_slices))
        results = es.scroll(scroll_id=results['_scroll_id'], scroll='35m')


def generate_docs(batch, index, model=None, nlp=None, arg_lexicon=None, progress_bar=True):
    json_input = not model

    if progress_bar:
        batch = tqdm(batch, desc='Preparing documents in batch', unit='docs', total=len(batch), leave=False)

    for doc in batch:
        doc_id = doc['_id'] if not json_input else doc.get('meta', {}).get('warc_id')

        if not doc_id:
            continue

        if doc.get('annotation_version', -1) >= ANNOTATION_VERSION:
            logger.error('{}: document annotation version greater or equal {}.'.format(doc_id, ANNOTATION_VERSION))
            continue

        output_doc = {}

        stats = defaultdict(lambda: {'num': 0, 'chars': 0, 'lines': 0, 'avg_len': 0.0})
        occurrences = defaultdict(lambda: 0)

        raw_text = doc.get('_source', {}).get('text_plain', '') if not json_input else doc.get('text', '')

        # Skip overly long texts to avoid running out of memory
        if len(raw_text) > 70000:
            continue

        if model:
            logger.info('Predicting lines...')
            lines = list(predict_raw_text(model, raw_text))
            labels = []
            start = 0
            end = 0
            prev_label = None
            for line in lines:
                if line[1] in ['<empty>', '<pad>']:
                    end += len(line[0])
                    continue

                if prev_label is None:
                    prev_label = line[1]

                if line[1] != prev_label:
                    labels.append((start, end, prev_label))
                    start = end

                prev_label = line[1]
                end += len(line[0])
            labels.append((start, end, prev_label))
        elif json_input:
            labels = doc.get('labels', [])
        else:
            raise RuntimeError('Neither model nor JSON input given')

        message_text = ''
        for start, end, label in labels:
            if label == 'paragraph':
                message_text += raw_text[start:end].strip() + '\n'

            stats[label]['num'] += 1
            stats[label]['chars'] += end - start
            stats[label]['lines'] += raw_text[start:end].count('\n')
            occurrences[label] += 1

        for label in stats:
            stats[label]['avg_len'] = stats[label]['chars'] / occurrences[label]

        stats['paragraph_quotation'] = {
            'num_ratio': (stats['paragraph']['num'] / stats['quotation']['num'])
            if stats['quotation']['num'] > 0 else -1,

            'lines_ratio': (stats['paragraph']['lines'] / stats['quotation']['lines'])
            if stats['quotation']['lines'] > 0 else -1,
        }

        output_doc['label_stats'] = dict(stats)

        # Extract arg lexicon classes
        if arg_lexicon:
            arg_classes = {}

            logger.info('Matching against arguing lexicon...')
            for regex, regex_text, cls in arg_lexicon:
                if cls in arg_classes:
                    continue
                if regex.search(message_text) is not None:
                    arg_classes[cls] = regex_text

            output_doc['arg_classes'] = list(arg_classes.keys())
            output_doc['arg_classes_matched_regex'] = list(arg_classes.values())

        # Improve language prediction by making use of content segmentation
        if len(message_text) > 15:
            logger.info('Detecting language')
            output_doc['lang'] = nlp(message_text)._.language['language']

        output_doc['annotation_version'] = ANNOTATION_VERSION
        output_doc['@modified'] = int(time() * 1000)

        yield {
            '_op_type': 'update',
            '_index': index,
            '_type': 'message',
            '_id': doc_id,
            'doc': output_doc
        }


def load_arglex(arglex_dir):
    """
    Load and parse Arguing Lexicon directory from
    https://mpqa.cs.pitt.edu/lexicons/arg_lexicon/ (Somasundaran et al., 2007)
    """
    files = glob(os.path.join(arglex_dir, '*.tff'))
    macros = {}
    rules = []

    for f in files:
        cls = None
        for line in open(f, 'r'):
            line = line.replace("\\'", "'").strip()

            if line.startswith('#class='):
                cls = line[8:len(line) - 1]
            elif line.startswith('#'):
                continue
            elif line.startswith('@'):
                m, v = line.split('=', 1)
                v = v[1:len(v) - 1].replace(', ', '|')
                macros[m] = v
            elif line.strip():
                rules.append((line.strip(), cls))

    rules_expanded = []
    for line, cls in rules:
        for m in macros:
            line = line.replace('({})'.format(m), '({})'.format(macros[m]))
        if line.strip():
            rules_expanded.append((line.strip(), cls))

    return rules_expanded


if __name__ == '__main__':
    main()
