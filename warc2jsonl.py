from bs4 import BeautifulSoup
import json
import mailparser
from warcio import ArchiveIterator


def main():
    with open('/home/roce3528/tmp/gmane/archives/news.gmane.org/gmane.ietf.nntp/000000.warc.gz', 'rb') as warc:
        with open('data/out.jsonl', 'w+') as jsonl:

            for i, record in enumerate(ArchiveIterator(warc)):
                mail = mailparser.parse_from_bytes(record.content_stream().read())
                if not mail.text_plain and mail.text_html:
                    text = BeautifulSoup('\n'.join(mail.text_html), 'html.parser').text
                else:
                    text = '\n'.join(mail.text_plain)

                print('Converting message {}'.format(i))
                json.dump({'text': text, 'external_id': record.rec_headers['WARC-Record-ID']}, jsonl)
                jsonl.write('\n')


if __name__ == '__main__':
    main()
