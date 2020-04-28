from glob import glob
import logging
import os
import re
import unicodedata

from elasticsearch import Elasticsearch, TransportError
import pyspark


def get_logger(name=''):
    """
    :param name: logger name
    :return: configured logging instance
    """
    logging.basicConfig(level=os.environ.get('LOGLEVEL', logging.WARN), format='%(levelname)s: %(message)s')
    logger = logging.getLogger(name)
    logger.setLevel(os.environ.get('LOGLEVEL', logging.INFO))
    return logger


def get_spark_context(app_name, job_desc=None):
    """
    :param app_name: Spark application name
    :param job_desc: Spark job description
    :return: Spark context
    """
    conf = pyspark.SparkConf()
    conf.setAppName(app_name)
    sc = pyspark.SparkContext(conf=conf)
    sc.setLogLevel('INFO')
    if job_desc:
        sc.setJobDescription(job_desc)
    return sc


def get_es_client():
    """
    :return: Configured Elasticsearch client
    """
    return Elasticsearch(['betaweb015', 'betaweb017', 'betaweb020'],
                         sniff_on_start=True, sniff_on_connection_fail=True, timeout=360)


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


def normalize_message_text(text):
    """
    Preprocess email plaintext for segmentation.

    :param text: email text
    :return: normalized message
    """
    if type(text) is bytes:
        text = text.decode('utf-8', 'ignore')

    if not text.strip():
        return text

    # Normalize email addresses
    text = re.sub(r'([a-zA-Z0-9_\-./+]+)@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.)|' +
                  r'(([a-zA-Z0-9\-]+\.)+))([a-zA-Z]{2,}|[0-9]{1,3})(\]?)', ' @EMAIL@ ', text)

    # Normalize URLs
    text = re.sub(r'[a-zA-Z]{3,5}://[\w.-]+(?:\.[\w.-]+)+[\w\-._~:/?#[\]@!$&\'()*+,;=]+', ' @URL@ ', text)

    # Normalize numbers
    text = re.sub(r'\d', '0', text)
    text = re.sub(r'0{5,}', '00000', text)

    # Normalize hash values
    text = re.sub(r'[0a-fA-F]{32,}', '@HASH@', text)

    # Preserve indents
    text = re.sub(r'(^|\n)[ \t]{4,}', r'\1@INDENT@ ', text)

    # Split off special characters
    text = re.sub(r'(^|[^<>|:.,;+=~!#*(){}\[\]])([<>|:.,;+=~!#*(){}\[\]]+)', r'\1 \2 ', text)

    # Truncate runs of special characters
    text = re.sub(r'([<>|:.,;+_=~\-!#*(){}\[\]]{5})[<>|:.,;+_=~\-!#*(){}\[\]]+', r'\1', text)

    # Normalize Unicode
    return unicodedata.normalize('NFKC', text)


def decode_message_part(message_part):
    """
    Decode email message MIME part with correct encoding.

    :param message_part: raw MIME part
    :return: decoded part contents
    """
    charset = message_part.get_content_charset()
    if charset is None or charset == '7-bit' or charset == '7bit':
        charset = 'us-ascii'
    elif charset == '8-bit' or charset == '8bit':
        charset = 'iso-8859-15'
    try:
        return message_part.get_payload(decode=True).decode(charset, errors='ignore')
    except LookupError:
        return message_part.get_payload(decode=True).decode('us-ascii', errors='ignore')


def get_message_id_prefix(message_id):
    """
    Get the prefix of a Gmane message ID that is most likely to match
    all references to a specific message.

    This is needed since Gmane message IDs and references to them are often not entirely identical.

    :param message_id: full message ID
    :return: ID prefix
    """
    s = message_id.split('@', maxsplit=1)
    if len(s) > 1 and s[1].startswith('public.gmane.org'):
        s_pre = s[0].split('-', maxsplit=1)[0]
        return s_pre if len(s_pre) > 7 and not s_pre.isdigit() else s[0]
    return s[0]


def retrieve_email_thread(es, index, message_id, restrict_to_same_group=True):
    """
    Recursively retrieve a full email thread based on message IDs.

    :param es: Elasticsearch client
    :param index: Elasticsearch index
    :param message_id: Message ID to use as a seed
    :param restrict_to_same_group: Restrict retrieval to messages from the same Gmane group
    :return: List of messages ordered by date
    """
    def create_should_clause(p):
        return [
            {'prefix': {'headers.message_id.keyword': p}},
            {'prefix': {'headers.in_reply_to.keyword': p}},
            {'prefix': {'headers.references.keyword': p}}
        ]

    retrieved_ids = set()
    id_prefix = get_message_id_prefix(message_id)

    must_clause = []
    must_not_clause = []
    should_clause = create_should_clause(id_prefix)
    query = {
        'query': {
            'bool': {
                'filter': {
                    'bool': {
                        'must': must_clause,
                        'must_not': must_not_clause,
                        'should': should_clause,
                        'minimum_should_match': 1
                    }
                }
            }
        },
        'sort': {
            '@timestamp': {'order': 'asc'}
        }
    }

    docs = []
    while True:
        if retrieved_ids:
            must_clause.clear()
            must_not_clause.append({'terms': {'headers.message_id.keyword': list(retrieved_ids)}})

        results = es.search(index=index, body=query, size=500)
        hits = results['hits']['hits']
        if not hits:
            break

        if not must_clause and restrict_to_same_group:
            must_clause.append({'term': {'groupname': hits[0]['_source']['groupname']}})

        references = set()
        for hit in hits:
            docs.append(hit)
            headers = hit['_source']['headers']

            if headers.get('message_id'):
                retrieved_ids.add(headers['message_id'])

            if headers.get('in_reply_to'):
                references.update(headers['in_reply_to']
                                  if type(headers['in_reply_to']) is list else [headers['in_reply_to']])

            if headers.get('references'):
                references.update(headers['references']
                                  if type(headers['references']) is list else [headers['references']])

        should_clause.clear()
        for message_id in (references - retrieved_ids):
            id_prefix = get_message_id_prefix(message_id)
            if id_prefix.rstrip():
                should_clause.extend(create_should_clause(id_prefix.rstrip()))

        if not should_clause:
            break

    return sorted(docs, key=lambda d: d['sort'])


def es_retry(func, *args, retries=3, **kwargs):
    """
    Call Elasticsearch function with parameters and return result.
    Logs errors and retries up to `retries` times in case of failure.

    :param func: Elasticsearch call
    :param retries: number of retries
    """
    for _ in range(retries):
        try:
            return func(*args, **kwargs)
        except TransportError as e:
            get_logger(__name__).error(e)
            pass
