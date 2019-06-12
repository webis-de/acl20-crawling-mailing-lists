from concurrent.futures import ThreadPoolExecutor
from glob import glob
import signal
import mailparser
from multiprocessing import Queue
import plac
import os
import psycopg2
import re
from time import sleep
from tqdm import tqdm
from threading import Thread
from warcio import ArchiveIterator


__SHUTDOWN_FLAG = False


@plac.annotations(
    input_dir=('Input directory containing newsgroup directories', 'positional', None, str, None, 'DIR'),
    database=('Output PostgreSQL database', 'option', 'd', str, None, 'DATABASE'),
    workers=('Number of Workers', 'option', 'w', int, None, 'NUM'),
)
def main(input_dir, database='warcs', workers=10):
    signal.signal(signal.SIGTERM, lambda s, f: signal_shutdown())
    signal.signal(signal.SIGINT, lambda s, f: signal_shutdown())

    indexer_thread = Thread(target=start_indexer, args=(input_dir, database, workers))
    indexer_thread.daemon = False
    indexer_thread.start()

    # Keep main thread responsive to catch signals
    while not __SHUTDOWN_FLAG:
        sleep(0.1)

    indexer_thread.join()


def start_indexer(input_dir, database, workers):
    with psycopg2.connect(database=database) as conn:
        print('Preparing database...')
        cur = conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS newsgroup (
            id SERIAL PRIMARY KEY NOT NULL,
            name VARCHAR(256) UNIQUE NOT NULL)""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS message (
            id BIGSERIAL PRIMARY KEY NOT NULL,
            message_id TEXT UNIQUE NOT NULL,
            newsgroup INTEGER REFERENCES newsgroup(id) NOT NULL,
            news_url TEXT NOT NULL,
            message_id_header TEXT NOT NULL,
            from_header TEXT NOT NULL,
            from_email TEXT NOT NULL,
            to_header TEXT NOT NULL,
            in_reply_to_header TEXT NOT NULL,
            message_date TIMESTAMP NOT NULL,
            filename TEXT NOT NULL,
            warc_offset INTEGER NOT NULL)""")
        conn.commit()

        print("Listing groups...")
        newsgroups = glob(os.path.join(input_dir, '*'))

        with tqdm(desc='Indexing group names', total=len(newsgroups), unit=' groups') as progress_bar:
            for i, group in enumerate(newsgroups):
                group = group.encode('utf-8', 'surrogateescape').decode('utf-8', 'ignore')
                cur.execute("""
                INSERT INTO newsgroup (name) VALUES (%s)
                    ON CONFLICT ON CONSTRAINT newsgroup_name_key DO UPDATE SET
                        name = excluded.name""", (os.path.basename(group),))
                if i > 0 and i % 500 == 0:
                    conn.commit()
                    progress_bar.update(500)
            progress_bar.update(progress_bar.total - progress_bar.n)
            conn.commit()

        cur.execute("""SELECT * FROM newsgroup""")
        newsgroups_dict = {n: i for i, n in cur}

        queue = Queue(maxsize=workers)
        with ThreadPoolExecutor(max_workers=workers) as e:
            with tqdm(desc='Indexing groups', total=len(newsgroups_dict), unit=' WARCs') as progress_bar:
                for group in newsgroups_dict:
                    for filename in glob(os.path.join(input_dir, group, '*.warc.gz')):
                        if __SHUTDOWN_FLAG:
                            return

                        queue.put(None)
                        e.submit(index_warc, conn, newsgroups_dict[group], filename, queue)

                    progress_bar.update()

                progress_bar.update(progress_bar.total - progress_bar.n)


def signal_shutdown():
    global __SHUTDOWN_FLAG
    __SHUTDOWN_FLAG = True


def index_warc(conn, newsgroup_id, filename, queue):
    global __SHUTDOWN_FLAG

    try:
        cur = conn.cursor()

        email_regex = re.compile(r'([a-zA-Z0-9_\-./+]+)@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.)|' +
                                 r'(([a-zA-Z0-9\-]+\.)+))([a-zA-Z]{2,}|[0-9]{1,3})(\]?)')

        with open(filename, 'rb') as f:
            iterator = ArchiveIterator(f)
            for i, record in enumerate(iterator):
                try:
                    headers = record.rec_headers
                    body = record.content_stream().read()
                    mail = mailparser.parse_from_bytes(body)

                    from_header = mail.headers.get('From', '')
                    from_email = re.search(email_regex, from_header)

                    cur.execute("""INSERT INTO message
                        (newsgroup, message_id, news_url, message_id_header, from_header, from_email, to_header,
                        in_reply_to_header, message_date, filename, warc_offset)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                                (newsgroup_id,
                                 headers.get_header('WARC-Record-ID'),
                                 headers.get_header('WARC-News-URL'),
                                 mail.headers.get('Message-ID', ''),
                                 from_header,
                                 from_email.group(0) if from_email is not None else '',
                                 mail.headers.get('To', ''),
                                 mail.headers.get('In-Reply-To', ''),
                                 str(mail.date) if mail.date is not None else '1970-01-01 00:00:00',
                                 os.path.realpath(filename),
                                 iterator.offset))
                    if i > 0 and i % 2000 == 0:
                        conn.commit()

                except psycopg2.DatabaseError as e:
                    print(e)
                    continue
                except Exception as e:
                    print(e)
                    continue

                if __SHUTDOWN_FLAG:
                    return
    finally:
        conn.commit()
        queue.get()


if __name__ == '__main__':
    plac.call(main)
