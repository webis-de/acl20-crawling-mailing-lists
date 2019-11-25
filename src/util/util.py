from elasticsearch import Elasticsearch
import pyspark
import re
import unicodedata


def get_spark_context(app_name, job_desc=None):
    conf = pyspark.SparkConf()
    conf.setMaster('yarn')
    conf.setAppName(app_name)
    sc = pyspark.SparkContext(conf=conf)
    sc.setLogLevel('INFO')
    if job_desc:
        sc.setJobDescription(job_desc)
    return sc


def get_es_client():
    return Elasticsearch(['betaweb015', 'betaweb017', 'betaweb020'],
                         sniff_on_start=True, sniff_on_connection_fail=True, timeout=360)


def normalize_message_text(text):
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
    charset = message_part.get_content_charset()
    if charset is None or charset == '7-bit' or charset == '7bit':
        charset = 'us-ascii'
    elif charset == '8-bit' or charset == '8bit':
        charset = 'iso-8859-15'
    try:
        return message_part.get_payload(decode=True).decode(charset, errors='ignore')
    except LookupError:
        return message_part.get_payload(decode=True).decode('us-ascii', errors='ignore')


def retrieve_email_thread(es, index, message_id):
    terms = [message_id]
    retrieved_ids = set()
    query = {
        'query': {
            'bool': {
                'filter': {
                    'bool': {
                        'should': [
                            {'terms': {'headers.message_id.keyword': terms}},
                            {'terms': {'headers.in_reply_to.keyword': terms}}
                        ]
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
            query['query']['bool']['must_not'] = {'terms': {'headers.message_id.keyword': list(retrieved_ids)}}

        results = es.search(index=index, body=query, size=500)
        if not results['hits']['hits']:
            break

        terms_temp = set()
        for hit in results['hits']['hits']:
            docs.append(hit)
            headers = hit['_source']['headers']

            if headers.get('message_id'):
                retrieved_ids.add(headers['message_id'])

            if headers.get('in_reply_to'):
                terms_temp.add(headers['in_reply_to'])

        terms.clear()
        terms.extend(terms_temp - retrieved_ids)

    return sorted(docs, key=lambda d: d['sort'])
