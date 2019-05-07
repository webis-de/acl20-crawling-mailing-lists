from concurrent.futures import ThreadPoolExecutor
from glob import glob
import signal
import mailparser
from multiprocessing import Queue
import plac
import os
import psycopg2
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
            name VARCHAR(255) UNIQUE NOT NULL)""")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS message (
            id BIGSERIAL PRIMARY KEY NOT NULL,
            message_id VARCHAR(255) UNIQUE NOT NULL,
            newsgroup INTEGER REFERENCES newsgroup(id) NOT NULL,
            news_url VARCHAR(255) NOT NULL,
            message_date TIMESTAMP NOT NULL,
            filename TEXT NOT NULL,
            warc_offset INTEGER NOT NULL)""")
        conn.commit()

        print("Listing groups...")
        newsgroups = glob(os.path.join(input_dir, '*'))

        with tqdm(desc='Indexing group names', total=len(newsgroups), unit=' groups') as progress_bar:
            for i, group in enumerate(newsgroups):
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
            with tqdm(desc='Indexing WARCs', total=len(newsgroups_dict), unit=' WARCs') as progress_bar:
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
    try:
        cur = conn.cursor()

        with open(filename, 'rb') as f:
            iterator = ArchiveIterator(f)
            for i, record in enumerate(iterator):
                headers = record.rec_headers
                body = record.content_stream().read()
                mail = mailparser.parse_from_bytes(body)
                cur.execute("""INSERT INTO message
                    (newsgroup, message_id, news_url, message_date, filename, warc_offset)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT ON CONSTRAINT message_id_key DO UPDATE SET
                        newsgroup = excluded.newsgroup,
                        message_id = excluded.message_id,
                        news_url = excluded.news_url,
                        message_date = excluded.message_date,
                        filename = excluded.filename,
                        warc_offset = excluded.warc_offset""",
                            (newsgroup_id,
                             headers.get_header('WARC-Record-ID'),
                             headers.get_header('WARC-News-URL'),
                             str(mail.date),
                             os.path.realpath(filename),
                             iterator.offset))
                if i > 0 and i % 2000 == 0:
                    conn.commit()

                if __SHUTDOWN_FLAG:
                    return
    finally:
        conn.commit()
        queue.get()


if __name__ == '__main__':
    plac.call(main)
