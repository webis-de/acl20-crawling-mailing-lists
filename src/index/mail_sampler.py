#!/usr/bin/env python3
#
# Draw a sample of messages from an Elasticsearch index.

from util.util import normalize_message_text
from elasticsearch import Elasticsearch
import json
import plac
from tqdm import tqdm


@plac.annotations(
    index=('Input Elasticsearch index', 'positional', None, str, None, 'INDEX'),
    output_jsonl=('Output JSONL file for import into Doccano', 'option', 'j', str, None, 'FILE'),
    output_text=('Output text file for learning FastText model', 'option', 't', str, None, 'FILE'),
    total_mails=('Total number of mails to sample', 'option', 'n', int, None, 'NUM'),
    group_limit=('Group sample limit', 'option', 'l', int, None, 'NUM'),
    skip=('Skip ahead n messages', 'option', 's', int, None, 'SKIP'),
    scroll_size=('Scroll size', 'option', 'x', int, None, 'SIZE')
)
def main(index, output_jsonl=None, output_text=None,
         total_mails=10000, group_limit=None, skip=0, scroll_size=2000):

    es = Elasticsearch(['betaweb015', 'betaweb017', 'betaweb020'],
                       sniff_on_start=True, sniff_on_connection_fail=True, timeout=120)

    jsonl_file = None
    if output_jsonl:
        jsonl_file = open(output_jsonl, 'w')

    text_file = None
    if output_text:
        text_file = open(output_text, 'w')

    print('Retrieving initial batch...')
    results = es.search(index=index, scroll='10m', size=scroll_size, body={
        "sort": ["warc_id"],
        "size": 200,
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
        }

        # "query": {
        #     "bool": {
        #         "must": [{
        #             "range": {
        #                 "label_stats.paragraph_quotation.num_ratio": {
        #                     "gte": 0.8,
        #                     "lte": 1.2
        #                 }
        #             }
        #         }, {
        #             "range": {
        #                 "label_stats.quotation.num": {
        #                     "gte": 5
        #                 }
        #             }
        #         }, {
        #             "term": {
        #                 "lang": {
        #                     "value": "en"
        #                 }
        #             }
        #         }],
        #         # "must_not": [{
        #         #     "wildcard": {
        #         #         "groupname": "gmane.ietf.*"
        #         #     }
        #         # }]
        #     }
        # }
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
                text_plain = src['text_plain']

                prev_samples = sampled_groups.get(src['groupname'], 0)
                if group_limit and prev_samples > group_limit:
                    continue
                sampled_groups[src['groupname']] = prev_samples + 1

                num_samples += 1
                progress_bar.update()

                if jsonl_file:
                    json.dump({'text': text_plain,
                               'meta': {k: src[k] for k in src.keys() if k not in ['text_plain', 'text_html']},
                               'labels': []}, jsonl_file)
                    jsonl_file.write('\n')

                if text_file:
                    text_file.write(normalize_message_text(text_plain))
                    text_file.write('\n')

                if num_samples >= total_mails:
                    break

            results = es.scroll(scroll_id=results['_scroll_id'], scroll='10m')

    if jsonl_file:
        jsonl_file.close()


if __name__ == '__main__':
    plac.call(main)
