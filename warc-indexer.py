from concurrent.futures import ThreadPoolExecutor
from elasticsearch import Elasticsearch, helpers
import email
import email.utils
from glob import glob
import signal
from multiprocessing import Queue
import plac
import pytz
import os
import re
import spacy
from spacy_langdetect import LanguageDetector
import sys
from time import sleep
from tqdm import tqdm
from threading import Thread
from util import decode_message_part
from warcio import ArchiveIterator


__SHUTDOWN_FLAG = False

ES = None
NLP = None


@plac.annotations(
    input_dir=('Input directory containing newsgroup directories', 'positional', None, str, None, 'DIR'),
    index=('Output Elasticsearch index', 'positional', None, str, None, 'INDEX'),
    workers=('Number of Workers', 'option', 'w', int, None, 'NUM'),
)
def main(input_dir, index, workers=10):
    signal.signal(signal.SIGTERM, lambda s, f: signal_shutdown())
    signal.signal(signal.SIGINT, lambda s, f: signal_shutdown())

    global ES, NLP
    ES = Elasticsearch(['betaweb015', 'betaweb017', 'betaweb020'], sniff_on_start=True, sniff_on_connection_fail=True, timeout=360)

    NLP = spacy.load('en_core_web_sm')
    NLP.add_pipe(LanguageDetector(), name='language_detector', last=True)

    indexer_thread = Thread(target=start_indexer, args=(input_dir, index, workers))
    indexer_thread.daemon = False
    indexer_thread.start()

    # Keep main thread responsive to catch signals
    while not __SHUTDOWN_FLAG:
        sleep(0.1)

    indexer_thread.join()


def start_indexer(input_dir, index, workers):
    if not ES.indices.exists(index=index):
        ES.indices.create(index=index, body={
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

    print("Listing groups...", file=sys.stderr)
    newsgroups = [os.path.basename(d) for d in glob(os.path.join(input_dir, 'gmane.*'))]

    print("Indexing group meta data...", file=sys.stderr)
    queue = Queue(maxsize=workers)
    with ThreadPoolExecutor(max_workers=workers) as e:
        with tqdm(desc='Index progress', total=len(newsgroups), unit=' groups') as progress_bar:
            for group in newsgroups:
                for filename in glob(os.path.join(input_dir, group, '*.warc.gz')):
                    if __SHUTDOWN_FLAG:
                        return

                    queue.put(None)
                    e.submit(index_warc, index, group, filename, queue)

                progress_bar.update()

            progress_bar.update(progress_bar.total - progress_bar.n)


def signal_shutdown():
    global __SHUTDOWN_FLAG
    __SHUTDOWN_FLAG = True


def index_warc(index, group, filename, queue):
    try:
        helpers.bulk(ES, generate_message(index, group, filename))
    except Exception as e:
        print(e, file=sys.stderr)
    finally:
        queue.get()


def generate_message(index, group, filename):
    global __SHUTDOWN_FLAG

    email_regex = re.compile(r'([a-zA-Z0-9_\-./+]+)@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.)|' +
                             r'(([a-zA-Z0-9\-]+\.)+))([a-zA-Z]{2,}|[0-9]{1,3})(\]?)')

    with open(filename, 'rb') as f:
        iterator = ArchiveIterator(f)
        for record in iterator:
            warc_headers = record.rec_headers
            body = record.content_stream().read()
            mail = email.message_from_bytes(body)

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
                lang = NLP(mail_text)._.language['language']
            except Exception as e:
                lang = 'UNKNOWN'
                print(e, file=sys.stderr)

            yield {
                '_index': index,
                '_type': 'message',
                '_id': warc_headers.get_header('WARC-Record-ID'),
                '_source': {
                    '@timestamp': mail_date,
                    'groupname': group,
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

            if __SHUTDOWN_FLAG:
                return


if __name__ == '__main__':
    plac.call(main)
