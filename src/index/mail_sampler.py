#!/usr/bin/env python3
#
# Draw a sample of messages from an Elasticsearch index.

import json

import click
from tqdm import tqdm

from util.util import normalize_message_text, get_es_client


@click.command()
@click.argument('index')
@click.option('-j', '--output-jsonl', help='Output JSONL file for import into Doccano', type=click.File('w'))
@click.option('-t', '--output-text', help='Output text file for learning FastText model', type=click.File('w'))
@click.option('-n', '--total-mails', help='Total number of mails to sample', type=int, default=1000)
@click.option('-l', '--group-limit', help='Group sample limit', type=int)
@click.option('-s', '--skip', help='Skip ahead n messages', type=int, default=0)
@click.option('-x', '--scroll-size', help='Scroll size', type=int, default=2000)
def main(index, output_jsonl, output_text,  total_mails, group_limit, skip, scroll_size):
    es = get_es_client()

    click.echo('Retrieving initial batch...', err=True)
    results = es.search(index=index, scroll='10m', size=scroll_size, body={
        "sort": ["warc_id"],
        "size": 200,
        "query": {
            "bool": {
                "filter": {
                    "bool": {
                        "must_not": [
                            {
                                "query_string": {
                                    "analyze_wildcard": True,
                                    "default_field": "*",
                                    "query": """groupname:(*.patches OR *.commits* OR
                                        *.dist-commits* OR *.version-control* OR *.git* OR *.cvs* OR *.svn* OR *.trunk*
                                        OR *.scm* OR *.pkg*) OR (groupname:(*.bugs* OR *.issues* OR *.bugzilla* OR
                                        *.codereview*) OR  headers.subject.keyword:(*jira* OR *bugzilla*) OR
                                        headers.from_email.keyword:(*bugs* OR *bugzilla* OR *jira* OR *jboss*))"""
                                }
                            }
                        ],
                        "must": {"term": {"lang": "en"}},
                        "minimum_should_match": 1,
                        "should": [
                            {"wildcard": {"groupname": "gmane.culture.*"}},
                            {"wildcard": {"groupname": "gmane.politics.*"}},
                            {"wildcard": {"groupname": "gmane.science.*"}},
                            {"wildcard": {"groupname": "gmane.education.*"}},
                            {"wildcard": {"groupname": "gmane.music.*"}},
                            {"wildcard": {"groupname": "gmane.games.*"}},
                            {"wildcard": {"groupname": "gmane.recreation.*"}}
                        ]
                    }
                }
            }
        }
    })

    if skip > 0:
        click.echo('Skipping ahead {} messages...'.format(skip), err=True)

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

                if output_jsonl:
                    json.dump({'text': text_plain,
                               'meta': {k: src[k] for k in src.keys() if k not in ['text_plain', 'text_html']},
                               'labels': []}, output_jsonl)
                    output_jsonl.write('\n')

                if output_text:
                    output_text.write(normalize_message_text(text_plain))
                    output_text.write('\n')

                if num_samples >= total_mails:
                    break

            results = es.scroll(scroll_id=results['_scroll_id'], scroll='10m')

    if output_jsonl:
        output_jsonl.close()


if __name__ == '__main__':
    main()
