#!/usr/bin/env python3

import csv
import email
import json
import sys

import click
from tqdm import tqdm

import util


@click.command()
@click.argument('input_file', type=click.File('r'))
@click.argument('output_file', type=click.File('w'))
def main(input_file, output_file):
    """
    Convert Kaggle Enron CSV to Doccano JSON format.
    """
    csv.field_size_limit(sys.maxsize)
    cvsreader = csv.reader(input_file)

    # skip header line
    next(cvsreader)

    for _, raw_message in tqdm(cvsreader, desc='Converting messages', unit=' messages'):
        mail = email.message_from_string(raw_message)
        mail_text = '\n'.join(util.decode_message_part(p) for p in mail.walk()
                              if p.get_content_type() == 'text/plain').strip()

        output_file.write(json.dumps({'text': mail_text, 'labels': []}) + '\n')

    output_file.close()


if __name__ == '__main__':
    main()
