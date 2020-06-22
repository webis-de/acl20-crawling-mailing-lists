#!/usr/bin/env python3

from functools import partial
from glob import glob
import os
import re
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


logger = util.get_logger(__name__)


@click.command()
@click.argument('input-dir', type=click.Path(exists=True, file_okay=False))
@click.argument('index')
def main(input_dir, index):
    """
    Index WARC files containing email/newsgroup messages to Elasticsearch.

    Arguments:
        input_dir: input directory containing raw WARC files
        index: Elasticsearch index
    """
    index_directory(input_dir, index)


def index_directory(input_dir, index):
    """
    Index WARC files from the given directory.

    :param input_dir: input directory containing raw WARC files
    :param index: Elasticsearch index
    """

    es = util.get_es_client()
    sc = util.get_spark_context('Mail WARC Indexer', 'Mail WARC Indexer for {}'.format(input_dir))

    if not es.indices.exists(index=index):
        es.indices.create(index=index, body={
            "settings": {
                "number_of_replicas": 0,
                "number_of_shards": 30
            },
            "mappings": {
                "properties": {
                    "modified": {"type": "date", "format": "epoch_millis"},
                    "headers": {
                        "properties": {
                            "date": {"type": "date", "format": "yyyy-MM-dd HH:mm:ssXXX"}
                        }
                    },
                    "id_hash": {"type": "long"},
                    "group": {"type": "keyword"},
                    "warc_file": {"type": "keyword"},
                    "warc_offset": {"type": "long"},
                    "warc_id": {"type": "keyword"},
                    "news_url": {"type": "keyword"},
                    "lang": {"type": "keyword"},
                    "text_plain": {"type": "text"},
                    "text_html": {"type": "text"}
                }
            }
        })

    counter = sc.accumulator(0)

    logger.info("Listing group directories")
    group_dirs = glob(os.path.join(input_dir, 'gmane.*'))
    group_dirs = sc.parallelize(group_dirs, len(group_dirs) // 5)

    logger.info('Listing WARCS')
    warcs = group_dirs.flatMap(lambda d: glob(os.path.join(d, '*.warc.gz')))
    warcs.cache()

    logger.info('Indexing messages')
    warcs.foreach(partial(_index_warc, index=index, counter=counter))


def _index_warc(filename, index, counter):
    """
    Index individual WARC file.

    :param filename: WARC file name
    :param index: Elasticsearch index
    :param counter: Spark counter
    """
    try:
        nlp = spacy.load('en_core_web_sm')
        nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)
        helpers.bulk(util.get_es_client(), _generate_docs(index, filename, nlp, counter))
    except Exception as e:
        logger.error(e)


def _generate_docs(index, filename, nlp, counter):
    """
    Generate Elasticsearch index docs.

    :param index: Elasticsearch index
    :param filename: WARC file name
    :param nlp: SpaCy language model
    :param counter: Spark counter
    :return: Generator of index doc actions
    """
    email_regex = re.compile(r'((?:[a-zA-Z0-9_\-./+]+)@(?:(?:\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.)|' +
                             r'(?:(?:[a-zA-Z0-9\-]+\.)+))(?:[a-zA-Z]{2,}|[0-9]{1,3})(?:\]?))')

    def split_header(header_name, header_dict, split_regex=','):
        headers = [re.sub(r'\s+', ' ', h).strip()
                   for h in re.split(split_regex, header_dict.get(header_name, '')) if h.strip()]
        if not headers:
            return None
        return headers if len(headers) > 1 else headers[0]

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
                # Convert offset outside +/-18:00 to UTC+0, since they would throw errors in Java's DateTime parser.
                if abs(d.utcoffset().total_seconds()) > 18 * 60 * 60:
                    d = d.astimezone(pytz.utc)
                mail_date = str(d)
            except TypeError:
                mail_date = None

            try:
                lang = nlp(mail_text[:nlp.max_length])._.language['language']
            except Exception as e:
                lang = 'UNKNOWN'
                logger.error(e)

            counter.add(1)

            yield {
                "_index": index,
                "_type": "message",
                "_id": doc_id,
                "_op_type": "update",
                "scripted_upsert": True,
                "script": {
                    "source": """
                        if (ctx._source.containsKey("lang")) {
                            params.doc.remove("lang");
                        }
                        ctx._source.putAll(params.doc);
                    """,
                    "params": {
                        "doc": {
                            "modified": int(time() * 1000),
                            "id_hash": hash(doc_id),
                            "group": os.path.basename(os.path.dirname(filename)),
                            "warc_file": os.path.join(os.path.basename(os.path.dirname(filename)),
                                                      os.path.basename(filename)),
                            "warc_offset": iterator.offset,
                            "warc_id": doc_id,
                            "news_url": warc_headers.get_header("WARC-News-URL"),
                            "headers": {
                                "date": mail_date,
                                "message_id": mail_headers.get("message-id"),
                                "from": from_header,
                                "from_email": from_email.group(0) if from_email is not None else "",
                                "subject": mail_headers.get("subject"),
                                "to": split_header("to", mail_headers),
                                "cc": split_header("cc", mail_headers),
                                "in_reply_to": split_header("in-reply-to", mail_headers),
                                "references": split_header("references", mail_headers, split_regex=r"\s"),
                                "list_id": mail_headers.get("list-id")
                            },
                            "lang": lang,
                            "text_plain": mail_text,
                            "text_html": mail_html
                        }
                    }
                },
                "upsert": {}
            }


if __name__ == '__main__':
    main()
