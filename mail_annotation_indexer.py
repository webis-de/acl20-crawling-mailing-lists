import ast
from elasticsearch import Elasticsearch, helpers
from collections import defaultdict
import json
import plac
import spacy
from spacy_langdetect import LanguageDetector
from tqdm import tqdm

ES = None

@plac.annotations(
    input_file=('Input annotation file', 'positional', None, str, None, 'FILE'),
    index=('Output Elasticsearch index', 'positional', None, str, None, 'INDEX')
)
def main(input_file, index):
    global ES, NLP
    ES = Elasticsearch(['betaweb015'], sniff_on_start=True, sniff_on_connection_fail=True, timeout=60)

    start_indexer(input_file, index)


def start_indexer(input_file, index):
    print('Creating index...')

    if not ES.indices.exists(index=index):
        ES.indices.create(index=index, body={
            'settings': {
                'number_of_replicas': 2,
                'number_of_shards': 5
            },
            'mappings': {
                'message': {
                    'properties': {
                        '@timestamp': {'type': 'date', 'format': 'yyyy-MM-dd HH:mm:ss'},

                        'groupname': {'type': 'keyword'},
                        'warc_file': {'type': 'keyword'},
                        'warc_offset': {'type': 'long'},
                        'warc_id': {'type': 'keyword'},
                        'news_url': {'type': 'keyword'},

                        'full_text': {'type': 'text'},
                        'lang': {'type': 'keyword'},

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
                }
            }
        })

    helpers.bulk(ES, generate_messages(index, input_file))


def generate_messages(index, input_file):
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)

    with tqdm(desc='Indexing messages', unit=' messages') as progress_bar:
        for line in open(input_file, 'r'):
            json_line = json.loads(line)

            source = {}
            source.update(json_line.get('meta', {}))
            source['full_text'] = json_line.get('text', '')
            if 'headers' in source and type(source['headers']) is str:
                source['headers'] = ast.literal_eval(source['headers'])

            stats = defaultdict(lambda: {'num': 0, 'chars': 0, 'lines': 0, 'avg_len': 0.0})
            occurrences = defaultdict(lambda: 0)

            for start, end, label in json_line.get('labels', []):
                stats[label]['num'] += 1
                stats[label]['chars'] += end - start
                stats[label]['lines'] += source['full_text'][start:end].count('\n')
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

            doc = nlp(source['full_text'])
            source['lang'] = doc._.language['language']

            source['label_stats'] = dict(stats)

            yield {
                '_index': index,
                '_type': 'message',
                '_id': None,
                '_source': source
            }

            progress_bar.update()


if __name__ == '__main__':
    plac.call(main)
