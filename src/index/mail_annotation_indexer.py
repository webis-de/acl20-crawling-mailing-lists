#!/usr/bin/env python3
#
# Index Doccano JSON annotations to Elasticsearch.
# The index is only updated and must exist.

from elasticsearch import Elasticsearch, helpers
from collections import defaultdict
import json
import plac
import spacy
from spacy_langdetect import LanguageDetector
from tqdm import tqdm


@plac.annotations(
    input_file=('Input annotation file', 'positional', None, str, None, 'FILE'),
    index=('Elasticsearch index', 'positional', None, str, None, 'INDEX')
)
def main(input_file, index):
    es = Elasticsearch(['betaweb015', 'betaweb017', 'betaweb020'],
                       sniff_on_start=True, sniff_on_connection_fail=True, timeout=120)
    start_indexer(es, input_file, index)


def start_indexer(es, input_file, index):
    if not es.indices.exists(index=index):
        raise RuntimeError('Index has to exist.')

    print('Loading language model...')
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)

    print('Updating mapping...')
    es.indices.put_mapping(index=index, doc_type='message', body={
                'properties': {
                    'label_stats': {
                        'properties': {
                            'paragraph_quotation.num_ratio': {'type': 'float'},
                            'paragraph_quotation.lines_ratio': {'type': 'float'}
                        }
                    }
                },
                "dynamic": True,
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

    helpers.bulk(es, generate_docs(index, input_file, nlp), timeout='240s')


def generate_docs(index, input_file, nlp):
    with tqdm(desc='Indexing annotations', unit=' messages') as progress_bar:
        for line in open(input_file, 'r'):
            json_line = json.loads(line)
            doc_id = json_line.get('meta', {}).get('warc_id')

            if not doc_id:
                continue

            index_doc = {}

            stats = defaultdict(lambda: {'num': 0, 'chars': 0, 'lines': 0, 'avg_len': 0.0})
            occurrences = defaultdict(lambda: 0)

            raw_text = json_line.get('text', '')
            message_text = ''
            for start, end, label in json_line.get('labels', []):
                if label == 'paragraph':
                    message_text += raw_text[start:end].strip() + '\n'

                stats[label]['num'] += 1
                stats[label]['chars'] += end - start
                stats[label]['lines'] += raw_text[start:end].count('\n')
                stats[label]['avg_len'] += end - start
                occurrences[label] += 1

            for label in stats:
                stats[label]['avg_len'] /= occurrences[label]

            stats['paragraph_quotation'] = {
                'num_ratio': (stats['paragraph']['num'] / stats['quotation']['num'])
                if stats['quotation']['num'] > 0 else -1,

                'lines_ratio': (stats['paragraph']['lines'] / stats['quotation']['lines'])
                if stats['quotation']['lines'] > 0 else -1,
            }

            index_doc['label_stats'] = dict(stats)

            # improve language prediction by making use of content segmentation
            if len(message_text) > 15:
                index_doc['lang'] = nlp(message_text)._.language['language']

            yield {
                '_op_type': 'update',
                '_index': index,
                '_type': 'message',
                '_id': doc_id,
                'doc': index_doc
            }

            progress_bar.update()


if __name__ == '__main__':
    plac.call(main)
