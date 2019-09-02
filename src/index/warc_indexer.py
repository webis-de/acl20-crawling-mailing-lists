#!/usr/bin/env python3
#
# Index WARC files containing email/newsgroup messages to Elasticsearch.

from elasticsearch import Elasticsearch, helpers
import email
import email.utils
from glob import glob
from functools import partial
import plac
import pytz
import os
import pyspark
import re
import spacy
from spacy_langdetect import LanguageDetector
import sys
from util.util import decode_message_part
from warcio import ArchiveIterator


@plac.annotations(
    input_dir=('Input directory containing newsgroup directories', 'positional', None, str, None, 'DIR'),
    index=('Output Elasticsearch index', 'positional', None, str, None, 'INDEX')
)
def main(input_dir, index):
    conf = pyspark.SparkConf()
    conf.setMaster('local[*]')
    conf.setAppName('Mail WARC Indexer')
    sc = pyspark.SparkContext(conf=conf)
    sc.setJobDescription('Mail WARC Indexer for {}'.format(input_dir))

    start_indexer(input_dir, index, sc)


def get_es_client():
    return Elasticsearch(['betaweb015', 'betaweb017', 'betaweb020'],
                         sniff_on_start=True, sniff_on_connection_fail=True, timeout=360)


def start_indexer(input_dir, index, spark_context):
    es = get_es_client()
    if not es.indices.exists(index=index):
        es.indices.create(index=index, body={
            'settings': {
                'number_of_replicas': 0,
                'number_of_shards': 30
            },
            'mappings': {
                'message': {
                    'properties': {
                        '@timestamp': {'type': 'date', 'format': 'yyyy-MM-dd HH:mm:ssZZ'},
                        'groupname': {'type': 'keyword'},
                        'warc_file': {'type': 'keyword'},
                        'warc_offset': {'type': 'long'},
                        'warc_id': {'type': 'keyword'},
                        'news_url': {'type': 'keyword'},
                        'lang': {'type': 'keyword'},
                        'text_plain': {'type': 'text'},
                        'text_html': {'type': 'text'}
                    }
                }
            }
        })

    counter = spark_context.accumulator(0)

    print("Listing group directories...", file=sys.stderr)
    group_dirs = glob(os.path.join(input_dir, 'gmane.*'))
    group_dirs = spark_context.parallelize(group_dirs, len(group_dirs) // 5)

    print('Listing WARCS...', file=sys.stderr)
    warcs = group_dirs.flatMap(lambda d: glob(os.path.join(d, '*.warc.gz')))
    warcs.cache()

    print('Indexing messages...', file=sys.stderr)
    warcs.foreach(partial(index_warc, index=index, counter=counter))


def index_warc(filename, index, counter):
    try:
        nlp = spacy.load('en_core_web_sm')
        nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)
        helpers.bulk(get_es_client(), generate_message(index, filename, nlp, counter))
    except Exception as e:
        print(e, file=sys.stderr)


def generate_message(index, filename, nlp, counter):
    email_regex = re.compile(r'([a-zA-Z0-9_\-./+]+)@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.)|' +
                             r'(([a-zA-Z0-9\-]+\.)+))([a-zA-Z]{2,}|[0-9]{1,3})(\]?)')

    with open(filename, 'rb') as f:
        iterator = ArchiveIterator(f)
        for record in iterator:
            warc_headers = record.rec_headers
            body = record.content_stream().read()
            mail = email.message_from_bytes(body)
            doc_id = warc_headers.get_header('WARC-Record-ID')

            mail_text = '\n'.join(decode_message_part(p) for p in mail.walk()
                                  if p.get_content_type() == 'text/plain').strip()
            mail_html = '\n'.join(decode_message_part(p) for p in mail.walk()
                                  if p.get_content_type() == 'text/html').strip()

            mail_headers = {h.lower(): str(mail[h]) for h in mail}
            from_header = mail_headers.get('from', '')
            from_email = re.search(email_regex, from_header)

            try:
                d = email.utils.parsedate_to_datetime(mail_headers.get('date'))
                if not d.tzinfo or d.tzinfo.utcoffset(d) is None:
                    d = pytz.utc.localize(d)
                mail_date = str(d)
            except TypeError:
                mail_date = None

            try:
                lang = nlp(mail_text[:nlp.max_length])._.language['language']
            except Exception as e:
                lang = 'UNKNOWN'
                print(e, file=sys.stderr)

            counter.add(1)

            yield {
                '_index': index,
                '_type': 'message',
                '_id': doc_id,
                '_op_type': 'update',
                'script': {
                    'source': 'ctx.noop = true'     # if document exists, do nothing
                },
                'upsert': {                         # upsert if document does not exist
                    '@timestamp': mail_date,
                    'groupname': os.path.basename(os.path.dirname(filename)),
                    'warc_file': os.path.join(os.path.basename(os.path.dirname(filename)),
                                              os.path.basename(filename)),
                    'warc_offset': iterator.offset,
                    'news_url': warc_headers.get_header('WARC-News-URL'),
                    'headers': {
                        'message_id': mail_headers.get('message-id'),
                        'from': from_header,
                        'from_email': from_email.group(0) if from_email is not None else '',
                        'subject': mail_headers.get('subject'),
                        'to': mail_headers.get('to'),
                        'cc': mail_headers.get('cc'),
                        'in_reply_to': mail_headers.get('in-reply-to'),
                        'list_id': mail_headers.get('list-id')
                    },
                    'lang': lang,
                    'text_plain': mail_text,
                    'text_html': mail_html
                }
            }


if __name__ == '__main__':
    plac.call(main)
