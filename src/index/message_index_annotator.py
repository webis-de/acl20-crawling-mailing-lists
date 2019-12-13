#!/usr/bin/env python3

from collections import defaultdict
import itertools
from functools import partial
import re
from time import time

from argument_esa_model.esa import ESA
import click
from elasticsearch import helpers
import spacy
from spacy_langdetect import LanguageDetector
from tensorflow.keras import models
from tqdm import tqdm

from parsing.message_segmenter import load_fasttext_model, predict_raw_text
from util import util


ANNOTATION_VERSION = 9

logger = util.get_logger(__name__)


@click.command()
@click.argument('index')
@click.argument('segmentation_model', type=click.Path(exists=True, dir_okay=False))
@click.argument('fasttext_model', type=click.Path(exists=True, dir_okay=False))
@click.option('-s', '--slices', help='Number of Elasticsearch scroll slices', type=int, default=100)
@click.option('-a', '--arg-lexicon', help='Arguing Lexicon directory (Somasundaran et al., 2007)',
              type=click.Path(exists=True, file_okay=False))
@click.option('-t', '--args-topic-model', multiple=True, type=click.Path(exists=True, dir_okay=False))
@click.option('-n', '--dry-run', help='Dry run (do not index anything)', is_flag=True)
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


def start_indexer(index, segmentation_model, fasttext_model, slices=1, **kwargs):
    """
    Start annotation indexer.

    :param index: Elasticsearch index
    :param segmentation_model: HDF5 Email Segmentation model
    :param fasttext_model: fastText email embedding
    :param slices: number of Elasticsearch scroll slices to process in parallel

    Keyword Args:
        dry_run (bool): Perform dry run, do not actually index anything
        progress_bar (bool): Show indexing progress bar
        arg_lexicon (str): Path to Arguing lexicon (Somasundaran, et al., 2007)
        args_topic_models (str): Args.me ESA topic model
    """

    if kwargs.get('dry_run'):
        logger.warning('Started in dry run mode, nothing will be indexed.')

    es = util.get_es_client()

    if not es.indices.exists(index=index):
        raise RuntimeError('Index has to exist.')

    logger.info('Updating Elasticsearch index mapping')
    es.indices.put_mapping(index=index, doc_type='message', body={
        'properties': {
            'main_content': {
                'type': 'text'
            },
            'segments': {
                'type': 'nested',
                'properties': {
                    'label': {
                        'type': 'keyword'
                    },
                    'begin': {
                        'type': 'integer'
                    },
                    'end': {
                        'type': 'integer'
                    }
                }
            },
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
            },
            'topics': {
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

    if kwargs.get('arg_lexicon'):
        logger.info('Loading Arguing Lexicon')
        kwargs['arg_lexicon'] = util.load_arglex(kwargs.get('arg_lexicon'))

    sc = util.get_spark_context('Mail Annotation Indexer')
    sc.range(0, slices).foreach(partial(_start_spark_worker,
                                        index=index, segmentation_model=segmentation_model,
                                        fasttext_model=fasttext_model, max_slices=slices, **kwargs))


def _start_spark_worker(slice_id, index, segmentation_model, fasttext_model, max_slices=1, **kwargs):
    logger.info('Loading SpaCy')
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)

    logger.info('Loading segmentation model')
    load_fasttext_model(fasttext_model)
    segmentation_model = models.load_model(segmentation_model)

    arg_lexicon = kwargs.get('arg_lexicon')
    if arg_lexicon:
        logger.info('Compiling arguing lexicon regex list')
        arg_lexicon = [(re.compile(r'\b' + regex + r'\b', re.IGNORECASE), regex, cls)
                       for regex, cls in arg_lexicon]

    topic_models = []
    if kwargs.get('args_topic_models'):
        for path in kwargs.get('args_topic_models'):
            topic_models.append(ESA(path))

    logger.info('Retrieving initial batch (slice {}/{})'.format(slice_id, max_slices))
    es = util.get_es_client()
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
        doc_gen = _generate_docs(results['hits']['hits'], index, segmentation_model, nlp,
                                 arg_lexicon=arg_lexicon, args_topic_models=topic_models, progress_bar=False)
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
        results = es.scroll(scroll_id=results['_scroll_id'], scroll='35m')


def _generate_docs(batch, index, segmentation_model, nlp, **kwargs):
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

    if kwargs.get('progress_bar'):
        batch = tqdm(batch, desc='Preparing documents in batch', unit='docs', total=len(batch), leave=False)

    for doc in batch:
        doc_id = doc['_id']

        if not doc_id:
            continue

        if doc.get('annotation_version', -1) >= ANNOTATION_VERSION:
            logger.error('{}: document annotation version greater or equal {}.'.format(doc_id, ANNOTATION_VERSION))
            continue

        output_doc = {}

        stats = defaultdict(lambda: {'num': 0, 'chars': 0, 'lines': 0, 'avg_len': 0.0})
        occurrences = defaultdict(lambda: 0)

        raw_text = doc.get('_source', {}).get('text_plain', '')

        # Skip overly long texts to avoid running out of memory
        if len(raw_text) > 70000:
            logger.warning('Skipping overly long message.')
            continue

        logger.debug('Segmenting message')
        lines = list(predict_raw_text(segmentation_model, raw_text))
        labels = []
        begin = 0
        end = 0
        prev_label = None
        for line in lines:
            if line[1] in ['<empty>', '<pad>']:
                end += len(line[0])
                continue

            if prev_label is None:
                prev_label = line[1]

            if line[1] != prev_label:
                labels.append((begin, end, prev_label))
                begin = end

            prev_label = line[1]
            end += len(line[0])
        labels.append((begin, end, prev_label))

        logger.debug('Calculating segment stats')
        main_content = ''
        output_doc['segments'] = []
        for begin, end, label in labels:
            if label in ['paragraph', 'section_heading']:
                main_content += raw_text[begin:end]

            stats[label]['num'] += 1
            stats[label]['chars'] += end - begin
            stats[label]['lines'] += raw_text[begin:end].count('\n')
            occurrences[label] += 1

            output_doc['segments'].append({'label': label, 'begin': begin, 'end': end})

        main_content = re.sub(r'\n{3,}', '\n\n', main_content).rstrip()

        # Remove any invalid surrogates
        main_content = main_content.encode('utf-8', 'replace').decode('utf-8')

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
        if kwargs.get('arg_lexicon'):
            arg_classes = {}

            logger.debug('Matching against arguing lexicon')
            for regex, regex_text, cls in kwargs.get('arg_lexicon'):
                if cls in arg_classes:
                    continue
                if regex.search(main_content) is not None:
                    arg_classes[cls] = regex_text

            output_doc['arg_classes'] = list(arg_classes.keys())
            output_doc['arg_classes_matched_regex'] = list(arg_classes.values())

        # Improve language prediction by making use of content segmentation
        if len(main_content) > 15:
            logger.debug('Detecting language')
            output_doc['lang'] = nlp(main_content)._.language['language']

        topics = set()
        if kwargs.get('args_topic_models'):
            logger.debug('Detecting topics')
            for tm in kwargs.get('args_topic_models'):
                topics.update([t for t, v in tm.process(main_content, False).items() if v >= 0.15])
        output_doc['topics'] = list(topics)

        output_doc['main_content'] = main_content
        output_doc['annotation_version'] = ANNOTATION_VERSION
        output_doc['@modified'] = int(time() * 1000)

        yield {
            '_op_type': 'update',
            '_index': index,
            '_type': 'message',
            '_id': doc_id,
            'doc': output_doc
        }


if __name__ == '__main__':
    main()
