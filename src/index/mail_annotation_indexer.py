#!/usr/bin/env python3
#
# Index Doccano JSON annotations to Elasticsearch.
# The index is only updated and must exist.

from parsing.message_segmenter import load_fasttext_model, predict_raw_text
from util.util import get_es_client

import click
from elasticsearch import helpers
from collections import defaultdict
import itertools
import json
import spacy
from spacy_langdetect import LanguageDetector
from tensorflow.python.keras import models
from time import time
from tqdm import tqdm

ANNOTATION_VERSION = 1


@click.command()
@click.argument('index')
@click.option('-m', '--model', help='Line segmentation model', type=click.Path(exists=True, dir_okay=False))
@click.option('-f', '--fmodel', help='Fasttext model', type=click.Path(exists=True, dir_okay=False))
@click.option('-i', '--input-file', help='Input annotation file', type=click.Path(exists=True, dir_okay=False))
def main(index, model, fasttext_model, input_file):
    if model and not fasttext_model:
        raise click.UsageError('FastText model is required if segmentation model is specified.')

    if not model and not input_file:
        raise click.UsageError('Need to specify either segmentation model or input annotation file.')

    start_indexer(index, model, fasttext_model, input_file)


def start_indexer(index, model, fasttext_model, input_file):
    click.echo('Updating mapping...', err=True)
    es = get_es_client()

    if not es.indices.exists(index=index):
        raise RuntimeError('Index has to exist.')

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

    click.echo('Loading SpaCy model...', err=True)
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)

    if model and fasttext_model:
        click.echo('Retrieving initial batch...', err=True)
        results = es.search(index=index, scroll='60m', size=200, body={
            'sort': ['_id'],
            'query': {
                'bool': {
                    'must_not': {
                        'range': {'annotation_version': {'gte': ANNOTATION_VERSION}}
                    }
                }
            }
        })

        click.echo('Loading models...', err=True)
        load_fasttext_model(fasttext_model)
        model = models.load_model(model)

        with tqdm(desc='Annotating and indexing batches', unit=' batches', total=float("inf")) as progress:
            while results['hits']['hits']:
                doc_gen = generate_docs(results['hits']['hits'], index, model, nlp)
                try:
                    # only start bulk request if generator has at least one element
                    peek = next(doc_gen)
                    helpers.bulk(es, itertools.chain([peek], doc_gen))
                except StopIteration:
                    pass
                progress.update(1)
                results = es.scroll(scroll_id=results['_scroll_id'], scroll='60m')

    elif input_file:
        click.echo('Indexing annotations...', err=True)
        helpers.bulk(es, generate_docs([json.loads(l) for l in open(input_file, 'r')], index, nlp=nlp))


def generate_docs(batch, index, model=None, nlp=None):
    json_input = not model

    with tqdm(desc='Preparing documents in batch', unit='docs', total=len(batch), leave=False) as progress:
        for doc in batch:
            doc_id = doc['_id'] if not json_input else doc.get('meta', {}).get('warc_id')

            if not doc_id:
                progress.update(1)
                continue

            output_doc = {}

            stats = defaultdict(lambda: {'num': 0, 'chars': 0, 'lines': 0, 'avg_len': 0.0})
            occurrences = defaultdict(lambda: 0)

            raw_text = doc.get('_source', {}).get('text_plain', '') if not json_input else doc.get('text', '')

            # Skip overly long texts to avoid running out of memory
            if len(raw_text) > 70000:
                progress.update(1)
                continue

            if model:
                lines = list(predict_raw_text(model, raw_text))
                labels = []
                start = 0
                end = 0
                prev_label = None
                for l in lines:
                    if l[1] in ['<empty>', '<pad>']:
                        end += len(l[0])
                        continue

                    if prev_label is None:
                        prev_label = l[1]

                    if l[1] != prev_label:
                        labels.append((start, end, prev_label))
                        start = end

                    prev_label = l[1]
                    end += len(l[0])
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

            # Improve language prediction by making use of content segmentation
            if len(message_text) > 15:
                output_doc['lang'] = nlp(message_text)._.language['language']

            output_doc['annotation_version'] = ANNOTATION_VERSION
            output_doc['@modified'] = int(time() * 1000)

            progress.update(1)

            yield {
                '_op_type': 'update',
                '_index': index,
                '_type': 'message',
                '_id': doc_id,
                'doc': output_doc
            }


if __name__ == '__main__':
    main()
