from util import *

from elasticsearch import Elasticsearch
import email
import json
import os
import plac
import re
from tqdm import tqdm
from warcio import ArchiveIterator


ES = None


@plac.annotations(
    index=('Input Elasticsearch index', 'positional', None, str, None, 'INDEX'),
    corpus_dir=('Input corpus directory', 'positional', None, str, None, 'CORPUS'),
    output_dir=('Output directory for eml files', 'option', 'd', str, None, 'DIR'),
    output_jsonl=('Output JSONL file for import into Doccano', 'option', 'j', str, None, 'FILE'),
    output_text=('Output text file for learning FastText model', 'option', 't', str, None, 'FILE'),
    total_mails=('Total number of mails to sample', 'option', 'n', int, None, 'NUM'),
    group_limit=('Group sample limit', 'option', 'l', int, None, 'NUM'),
    skip=('Skip ahead n messages', 'option', 's', int, None, 'SKIP')
)
def main(index, corpus_dir, output_dir=None, output_jsonl=None, output_text=None,
         total_mails=10000, group_limit=None, skip=0):
    scroll_size = 2000

    global ES
    ES = Elasticsearch(['betaweb015'], sniff_on_start=True, sniff_on_connection_fail=True, timeout=60)

    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    jsonl_file = None
    if output_jsonl:
        jsonl_file = open(output_jsonl, 'w')

    text_file = None
    if output_text:
        text_file = open(output_text, 'w')

    print('Retrieving initial batch...')
    results = ES.search(index=index, scroll='3m', size=scroll_size, body={
        "query": {
            "bool": {
                "filter": {
                    # Skip Gwene
                    "wildcard": {"groupname": "gmane.*"}
                },
                "must_not": [{
                    # SCM
                    "query_string": {
                        "query": "groupname:(*.patches OR *.commits* OR *.dist-commits* OR *.version-control* " +
                                 "OR *.git* OR *.cvs* OR *.svn* OR *.trunk* OR *.scm* OR *.pkg*)",
                        "analyze_wildcard": True
                    }
                }, {
                    # Bugs
                    "query_string": {
                        "query": "groupname:(*.bugs* OR *.issues* OR *.bugzilla* OR *.codereview*)",
                        "analyze_wildcard": True
                    }
                }, {
                    # Comp, Linux, OS, User
                    "query_string": {
                        "query": "groupname:(gmane.comp.* OR gmane.linux.* OR gmane.os.* OR *.user)",
                        "analyze_wildcard": True
                    }
                }]
            }
        },
        "sort": ["warc_id"]
    })

    if skip > 0:
        print('Skipping ahead {} messages...'.format(skip))

    sampled_groups = {}
    num_samples = 0
    num_skipped = 0

    with tqdm(desc='Calculating progress', unit=' messages') as progress_bar:
        while num_samples < total_mails and len(results['hits']['hits']) > 0:
            for hit in results['hits']['hits']:
                if skip > 0 and num_skipped < skip:
                    progress_bar.set_description('Skipping messages')
                    progress_bar.total = skip
                    num_skipped += 1
                    progress_bar.update()
                    continue
                elif (skip == 0 or num_skipped >= skip) and num_samples == 0:
                    progress_bar.set_description('Sampling messages')
                    progress_bar.total = total_mails
                    progress_bar.n = 0
                    progress_bar.last_print_n = 0
                    progress_bar.update(0)

                src = hit['_source']
                prev_samples = sampled_groups.get(src['groupname'], 0)
                if group_limit and prev_samples > group_limit:
                    continue
                sampled_groups[src['groupname']] = prev_samples + 1

                with open(os.path.join(corpus_dir, src['warc_file']), 'rb') as f:
                    f.seek(src['warc_offset'])
                    record = next(ArchiveIterator(f))

                    msg_url = record.rec_headers.get_header('WARC-News-URL')

                    payload = record.content_stream().read()
                    payload_str = '\n'.join(decode_message_part(p) for p in email.message_from_bytes(payload).walk()
                                            if p.get_content_type() == 'text/plain').strip()

                    # skip empty or binary payload
                    if not payload_str or re.match(r'^\s*=\s*ybegin\s', payload_str):
                        continue

                    num_samples += 1
                    progress_bar.update()

                    if output_dir:
                        output_file = os.path.join(output_dir, msg_url.replace('news:', '').replace('/', '') + '.eml')
                        open(output_file, 'wb').write(payload + b'\n')

                    if jsonl_file:
                        json.dump({'text': payload_str,
                                   'meta': {k: str(src[k]) for k in src.keys()}, 'labels': []}, jsonl_file)
                        jsonl_file.write('\n')

                    if text_file:
                        text_file.write(normalize_message_text(payload_str))
                        text_file.write('\n')

                if num_samples >= total_mails:
                    break

            results = ES.scroll(scroll_id=results['_scroll_id'], scroll='3m')

    if jsonl_file:
        jsonl_file.close()


if __name__ == '__main__':
    plac.call(main)
