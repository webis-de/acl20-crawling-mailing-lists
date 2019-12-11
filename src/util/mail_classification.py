from collections import deque
import multiprocessing
import json

import fastText
import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.python.client import device_lib

from util import util

logger = util.get_logger(__name__)


_LABEL_MAP = {
    'paragraph': 0,
    'closing': 1,
    'inline_headers': 2,
    'log_data': 3,
    'mua_signature': 4,
    'patch': 5,
    'personal_signature': 6,
    'quotation': 7,
    'quotation_marker': 8,
    'raw_code': 9,
    'salutation': 10,
    'section_heading': 11,
    'tabular': 12,
    'technical': 13,
    'visual_separator': 14,
    '<empty>': 15
}


class MailLinesSequence(Sequence):
    """Keras Sequence of email message lines."""

    LABEL_MAP = _LABEL_MAP
    LABEL_MAP_INVERSE = {_LABEL_MAP[k]: k for k in _LABEL_MAP}
    LABEL_MAP_ONEHOT = {label: onehot for label, onehot in zip(_LABEL_MAP, np.eye(len(_LABEL_MAP)))}

    def __init__(self, input_data, context_shape, labeled=True, batch_size=None,
                 input_is_raw_text=False, max_lines=None):
        """
        :param input_data: input JSON file (file handle or path) with training data or raw email text
        :param context_shape: shape of the context window (2*context+1, line_len, word_dim)
        :param labeled: whether input contains labels
        :param batch_size: mini-batch size
        :param input_is_raw_text: whether input is a file or a raw email string
        :param max_lines: maximum number of lines to load from the input source (rest is discarded)
        """
        self.labeled = labeled
        self.mail_lines = []
        self.mail_start_indices = set()
        self.mail_end_indices = set()
        self.mail_metadata_map = {}

        self.batch_size = batch_size
        self.line_shape = context_shape[1:]
        self.context_size = context_shape[0] // 2
        self.output_dim = len(self.LABEL_MAP)

        if not input_is_raw_text:
            if type(input_data) is str:
                self._load_jsonl(open(input_data, 'r'), max_lines)
            else:
                self._load_jsonl(input_data, max_lines)
        else:
            self._load_raw_text(input_data, max_lines)

    def _load_jsonl(self, json_file, max_lines):
        """
        Load data from JSON file.

        :param json_file: path or file handle
        :param max_lines: maximum number of lines to load from the file (rest is discarded).
        """
        for i, json_text in enumerate(json_file):
            mail_json = json.loads(json_text)

            lines = None
            if not self.labeled:
                lines = [l + '\n' for l in mail_json['text'].split('\n')]

            elif self.labeled and mail_json['labels']:
                lines = [l for l in annotation_dict_to_lines(mail_json)]

            # Skip overly long mails (probably just excessive log data)
            if len(lines) > 5000:
                continue

            if lines:
                self.mail_start_indices.add(len(self.mail_lines))
                self.mail_metadata_map[len(self.mail_lines)] = mail_json
                self.mail_lines.extend(lines)
                self.mail_end_indices.add(len(self.mail_lines))

            if max_lines is not None and i >= max_lines:
                break

        if self.batch_size is None:
            self.batch_size = len(self.mail_lines)

    def _load_raw_text(self, raw_text, max_lines):
        """
        Split raw text into lines

        :param raw_text: input text
        :param max_lines: maximum number of lines to load from the text (rest is discarded).
        """
        if max_lines is not None:
            lines = [l + '\n' for l in raw_text.split('\n')[:max_lines]]
        else:
            lines = [l + '\n' for l in raw_text.split('\n')]

        if lines:
            self.mail_start_indices.add(len(self.mail_lines))
            self.mail_lines.extend(lines)
            self.mail_end_indices.add(len(self.mail_lines))

        if self.batch_size is None:
            self.batch_size = len(self.mail_lines)

    def __len__(self):
        return int(np.ceil(len(self.mail_lines) / self.batch_size))

    def __getitem__(self, index):
        index = index * self.batch_size
        end_index = index + self.batch_size if self.batch_size is not None else len(self.mail_lines)
        end_index = min(end_index, len(self.mail_lines))

        padding_line = np.ones(self.line_shape)
        batch = np.empty((self.batch_size,) + self.line_shape)
        batch_prev = np.empty((self.batch_size,) + self.line_shape)
        batch_context = np.empty((self.batch_size, self.context_size * 2 + 1) + self.line_shape)
        batch_labels = np.empty((self.batch_size, self.output_dim))

        mail_slice = self.mail_lines[index:end_index]

        def _get_line(line):
            # Strip labels from line
            return line if not self.labeled else line[0]

        for i, line in enumerate(mail_slice):
            if self.labeled:
                batch_labels[i] = line[1]

            context_lines = deque()

            # Assemble previous context with padding
            while len(context_lines) < self.context_size:
                ci = index + i - len(context_lines) - 1
                if ci < 0 or ci + 1 in self.mail_start_indices:
                    context_lines.extendleft([padding_line] * (self.context_size - len(context_lines)))
                    break
                word_vecs = get_word_vectors(_get_line(self.mail_lines[ci]))
                context_lines.appendleft(self._pad_line_vectors(word_vecs, self.line_shape[0]))

            # Add current line to context
            word_vecs = get_word_vectors(_get_line(line))
            context_lines.append(self._pad_line_vectors(word_vecs, self.line_shape[0]))

            # Assemble following context with padding
            while len(context_lines) < 2 * self.context_size + 1:
                ci = index + i + len(context_lines)
                if ci >= len(self.mail_lines) or ci in self.mail_end_indices:
                    context_lines.extend([padding_line] * ((2 * self.context_size + 1) - len(context_lines)))
                    break
                word_vecs = get_word_vectors(_get_line(self.mail_lines[ci]))
                context_lines.append(self._pad_line_vectors(word_vecs, self.line_shape[0]))

            batch[i] = context_lines[self.context_size]
            batch_prev[i] = context_lines[self.context_size - 1]
            batch_context[i] = np.stack(context_lines)

        if self.labeled:
            return [batch, batch_prev, batch_context], batch_labels

        return [batch, batch_prev, batch_context]

    @staticmethod
    def _pad_line_vectors(vectors, max_len):
        """
        Assemble a list of variable-length vectors into a padded 2D matrix of dimensions (n, max_len).

        :param vectors: list or array of vectors
        :param max_len: maximum line length
        :return: padded matrix
        """
        if vectors.shape[0] > max_len:
            pivot_idx = int(np.ceil(max_len * .75))
            vectors = np.concatenate((vectors[:pivot_idx], vectors[vectors.shape[0] - max_len + pivot_idx:]))

        return np.pad(vectors, ((0, max(0, max_len - vectors.shape[0])), (0, 0)), 'constant')

    _fasttext_model = None

    @property
    def num_workers(self):
        """Get appropriate number of multiprocessing data workers to use with this MailLinesSequence."""
        return multiprocessing.cpu_count() if has_gpu() else 2

    @property
    def max_queue_size(self):
        """Get appropriate maximum queue size to use with this MailLinesSequence."""
        return 10 if has_gpu() else 200


def has_gpu():
    """
    :return: whether a GPU device is available
    """
    return 'GPU' in str(device_lib.list_local_devices())


def annotation_dict_to_lines(annotation_doc):
    """
    Parse Annotations in Doccano JSON format to one-hot labeled lines.

    :param annotation_doc: input dict
    :return: Generator of (line text, label as one-hot)
    """
    lines = [l + '\n' for l in annotation_doc['text'].split('\n')]
    annotations = sorted(get_annotations_from_dict(annotation_doc), key=lambda a: a['start_offset'], reverse=True)
    offset = 0
    for l in lines:
        end_offset = offset + len(l)

        if annotations and offset > annotations[-1]['end_offset']:
            annotations.pop()

        if not annotations or not l.strip():
            yield l, MailLinesSequence.LABEL_MAP_ONEHOT['<empty>']
            offset = end_offset
            continue

        if offset < annotations[-1]['end_offset'] and end_offset > annotations[-1]['start_offset']:
            yield l, MailLinesSequence.LABEL_MAP_ONEHOT[annotations[-1]['label']]
        else:
            yield l, MailLinesSequence.LABEL_MAP_ONEHOT['<empty>']

        offset = end_offset


def get_annotations_from_dict(d):
    """
    Get annotations from Doccano either one of the two Doccano export formats.
    This is a little compatibility hack make both formats work.
    """
    if 'annotations' in d:
        return d['annotations']
    elif 'labels' in d:
        return [{'start_offset': a[0], 'end_offset': a[1], 'label': a[2]} for a in d['labels']]


_fasttext_model = None


def load_fasttext_model(model_path):
    """
    Load trained fastText model from given path and cache it in memory.
    This function has to be called only once. Calling it multiple times will do nothing.

    :param model_path: path to fastText model
    """
    global _fasttext_model
    if not _fasttext_model:
        _fasttext_model = fastText.load_model(model_path)


def get_word_vectors(text):
    """
    Tokenize text and return fastText word vectors.
    Requires a fastText model to be loaded (see :func:`load_fasttext_model`)

    :param text: input text
    :return: word vector matrix
    """
    try:
        matrix = [_fasttext_model.get_word_vector(w) for w in fastText.tokenize(util.normalize_message_text(text))]
    except Exception as e:
        logger.error('Failed to tokenize line: {}'.format(e))
        matrix = [get_word_vector('')]

    if len(matrix) == 0:
        matrix = [get_word_vector('')]

    return np.array(matrix)


def get_word_vector(word):
    """
    Get fastText embedding for individual word.
    Requires a fastText model to be loaded (see :func:`load_fasttext_model`)

    :param word: input word token
    :return: word vector
    """
    if _fasttext_model is None:
        raise RuntimeError("FastText vectors not loaded. Call load_fasttext_model() first.")

    try:
        return _fasttext_model.get_word_vector(word)
    except Exception as e:
        logger.error('Failed to obtain word vector: {}'.format(e))
        return _fasttext_model.get_word_vector('')
