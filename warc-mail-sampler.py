from mail_cleanup_deep import normalize_fasttext_input

import email
import json
import os
import plac
import psycopg2
import psycopg2.extras
from warcio import ArchiveIterator


@plac.annotations(
    output_dir=('Output directory for eml files', 'option', 'd', str, None, 'DIR'),
    output_jsonl=('Output JSONL file for import into Doccano', 'option', 'j', str, None, 'FILE'),
    output_text=('Output text file for learning FastText model', 'option', 't', str, None, 'FILE'),
    database=('Input PostgreSQL database', 'option', 'i', str, None, 'DATABASE'),
    total_mails=('Total number of mails to sample', 'option', 'n', int, None, 'NUM'),
    group_limit=('Group sample limit', 'option', 'l', int, None, 'NUM')
)
def main(output_dir=None, output_jsonl=None, output_text=None, database='warcs', total_mails=10000, group_limit=80):
    with psycopg2.connect(database=database) as conn:
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor, name='warc_samples')
        cur.itersize = 2000

        sampled_groups = {}
        num_samples = 0

        print('Retrieving samples...')

        cur.execute("""
            SELECT * FROM message
            INNER JOIN newsgroup n ON message.newsgroup = n.id
            WHERE n.name NOT LIKE 'gwene.%%'
                AND n.name NOT LIKE '%%.patches%%'
                AND n.name NOT LIKE '%%.commits%%'
                AND n.name NOT LIKE '%%.cvs%%'
                AND n.name NOT LIKE '%%.svn%%'
            ORDER BY RANDOM()""")

        if output_dir and not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        jsonl_file = None
        if output_jsonl:
            jsonl_file = open(output_jsonl, 'w')

        text_file = None
        if output_text:
            text_file = open(output_text, 'w')

        for msg in cur:
            prev_samples = sampled_groups.get(msg['name'], 0)
            if prev_samples > group_limit:
                continue
            sampled_groups[msg['name']] = prev_samples + 1
            num_samples += 1

            with open(msg['filename'], 'rb') as f:
                f.seek(msg['warc_offset'])
                record = next(ArchiveIterator(f))

                msg_url = record.rec_headers.get_header('WARC-News-URL')
                print('Sampled {}'.format(msg_url))

                payload = record.content_stream().read()
                payload_str = '\n'.join(decode_message_part(p) for p in email.message_from_bytes(payload).walk()
                                        if p.get_content_type() == 'text/plain')
                if not payload_str:
                    continue

                if output_dir:
                    output_file = os.path.join(output_dir, msg_url.replace('news:', '').replace('/', '') + '.eml')
                    open(output_file, 'wb').write(payload + b'\n')

                if jsonl_file:
                    json.dump({'text': payload_str,
                               'meta': {k: str(msg[k]) for k in msg.keys()}, 'labels': []}, jsonl_file)
                    jsonl_file.write('\n')

                if text_file:
                    text_file.write(normalize_fasttext_input(payload_str))
                    text_file.write('\n')

            if num_samples >= total_mails:
                break

        if jsonl_file:
            jsonl_file.close()


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


if __name__ == '__main__':
    plac.call(main)
