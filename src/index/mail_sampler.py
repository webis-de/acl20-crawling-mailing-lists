#!/usr/bin/env python3

import json

import click
from tqdm import tqdm

from util import util


logger = util.get_logger(__name__)


@click.command()
@click.argument('index')
@click.argument('output_file')
@click.option('-q', '--query', help='Sample selection query', type=click.File('r'))
@click.option('-f', '--output-format', help='Output format (multiple may be specified)', multiple=True,
              type=click.Choice(['json', 'text']), default='json')
@click.option('-n', '--total-mails', help='Total number of mails to sample', type=int, default=1000)
@click.option('-l', '--group-limit', help='Group sample limit', type=int)
@click.option('-s', '--skip', help='Skip ahead n messages', type=int, default=0)
@click.option('-x', '--scroll-size', help='Scroll size', type=int, default=2000)
def main(index, output_file, **kwargs):
    """
    Sample mails from Elasticsearch index.

    Arguments:
        index: Elasticsearch index to sample from
        output_file: output file (prefix without extension in case multiple formats are specified)
    """

    output_jsonl = None
    output_text = None
    if 'json' in kwargs['output_format']:
        fname = output_file if len(kwargs['output_format']) == 1 else kwargs['output_format'] + '.jsonl'
        output_jsonl = open(fname, 'w')
    if 'text' in kwargs['output_format']:
        fname = output_file if len(kwargs['output_format']) == 1 else kwargs['output_format'] + '.txt'
        output_text = open(fname, 'w')

    if kwargs.get('query') is not None:
        query = json.load(kwargs.get('query'))
    else:
        query = {
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
                                            *.dist-commits* OR *.version-control* OR *.git* OR *.cvs* OR *.svn*
                                            OR *.trunk* OR *.scm* OR *.pkg*) OR (groupname:(*.bugs* OR *.issues*
                                            OR *.bugzilla* OR *.codereview*) OR 
                                            headers.subject.keyword:(*jira* OR *bugzilla*) OR
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
        }

    logger.info('Retrieving initial batch')
    es = util.get_es_client()
    results = util.es_retry(es.search, index=index, scroll='10m', size=kwargs['scroll_size'], body=query)

    skip = kwargs['skip']
    if skip > 0:
        logger.info('Skipping ahead {} messages'.format(skip))

    sampled_groups = {}
    num_samples = 0
    num_skipped = 0

    with tqdm(desc='Calculating progress', unit=' messages') as progress_bar:
        while num_samples < kwargs['total_mails'] and len(results['hits']['hits']) > 0:
            for hit in results['hits']['hits']:
                if skip > 0 and num_skipped < skip:
                    progress_bar.set_description('Skipping messages')
                    progress_bar.total = skip
                    num_skipped += 1
                    progress_bar.update()
                    continue
                elif (skip == 0 or num_skipped >= skip) and num_samples == 0:
                    progress_bar.set_description('Sampling messages')
                    progress_bar.total = kwargs['total_mails']
                    progress_bar.n = 0
                    progress_bar.last_print_n = 0
                    progress_bar.update(0)

                src = hit['_source']
                text_plain = src['text_plain']

                prev_samples = sampled_groups.get(src['groupname'], 0)
                if kwargs['group_limit'] and prev_samples > kwargs['group_limit']:
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
                    output_text.write(util.normalize_message_text(text_plain))
                    output_text.write('\n')

                if num_samples >= kwargs['total_mails']:
                    break

            results = util.es_retry(es.scroll, scroll_id=results['_scroll_id'], scroll='10m')

    if output_jsonl:
        output_jsonl.close()
    if output_text:
        output_text.close()


if __name__ == '__main__':
    main()
