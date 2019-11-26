#!/usr/bin/env python3
#
# Index WARC files containing email/newsgroup messages to Elasticsearch.

from functools import partial
from glob import glob
import os
import re
import sys
from time import time

import click
from elasticsearch import helpers
import email
import email.utils
import pytz
import spacy
from spacy_langdetect import LanguageDetector
from warcio import ArchiveIterator

from util import util


@click.command()
@click.argument('input-dir', type=click.Path(exists=True, file_okay=False))
@click.argument('index')
def main(input_dir, index):
    start_indexer(input_dir, index)


def start_indexer(input_dir, index):
    es = util.get_es_client()
    sc = util.get_spark_context('Mail WARC Indexer', 'Mail WARC Indexer for {}'.format(input_dir))

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
                        '@modified': {'type': 'date', 'format': 'epoch_millis'},
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

    counter = sc.accumulator(0)

    print("Listing group directories...", file=sys.stderr)
    group_dirs = glob(os.path.join(input_dir, 'gmane.*'))
    group_dirs = sc.parallelize(group_dirs, len(group_dirs) // 5)

    print('Listing WARCS...', file=sys.stderr)
    warcs = group_dirs.flatMap(lambda d: glob(os.path.join(d, '*.warc.gz')))
    warcs.cache()

    print('Indexing messages...', file=sys.stderr)
    warcs.foreach(partial(index_warc, index=index, counter=counter))


def index_warc(filename, index, counter):
    try:
        nlp = spacy.load('en_core_web_sm')
        nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)
        helpers.bulk(util.get_es_client(), generate_message(index, filename, nlp, counter))
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

            mail_text = '\n'.join(util.decode_message_part(p) for p in mail.walk()
                                  if p.get_content_type() == 'text/plain').strip()
            mail_html = '\n'.join(util.decode_message_part(p) for p in mail.walk()
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
                'scripted_upsert': True,
                'script': {
                    'source': '''
                        if (ctx._source.containsKey("lang")) {
                            params.doc.remove("lang");
                        }
                        ctx._source.putAll(params.doc);                    
                    ''',
                    'params': {
                        'doc': {
                            '@timestamp': mail_date,
                            '@modified': int(time() * 1000),
                            'groupname': os.path.basename(os.path.dirname(filename)),
                            'warc_file': os.path.join(os.path.basename(os.path.dirname(filename)),
                                                      os.path.basename(filename)),
                            'warc_offset': iterator.offset,
                            'warc_id': doc_id,
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
                },
                'upsert': {}
            }


if __name__ == '__main__':
    main()
