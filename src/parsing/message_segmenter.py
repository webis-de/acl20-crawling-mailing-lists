#!/usr/bin/env python3


from datetime import datetime
from itertools import chain
import json
import logging
import os
import re

import click
import fastText
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks, layers, metrics, models
from tensorflow.keras.utils import Sequence
from tqdm import tqdm
import numpy as np

from util import util

logger = logging.getLogger(__name__)

label_map = {
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
    '<empty>': 15,
    '<pad>': 16
}

label_map_inverse = {label_map[k]: k for k in label_map}
label_map_onehot = {label: onehot for label, onehot in zip(label_map, np.eye(len(label_map)))}

INPUT_DIM = 100
OUTPUT_DIM = len(label_map_onehot)
BATCH_SIZE = 128
MAX_LEN = 12
CONTEXT = 4


@click.group()
def main():
    pass


@main.command()
@click.argument('fasttext-model', type=click.Path(exists=True, dir_okay=False))
@click.argument('train-data', type=click.Path(exists=True, dir_okay=False))
@click.argument('output', type=click.Path(exists=False))
@click.option('-v', '--validation-data', help='Validation Data JSON')
def train(fasttext_model, train_data, output, validation_data):
    """
    Train message segmenter to classify lines of an email or newsgroup message.

    Arguments:
        FASTTEXT_MODEL: pre-trained FastText embedding
        TRAIN_DATA: input training data as JSON
        OUTPUT: Model output
    """
    logger.info('Loading FastText model...')
    load_fasttext_model(fasttext_model)
    train_model(train_data, output, validation_data)


@main.command()
@click.argument('model', type=click.Path(exists=True, dir_okay=False))
@click.argument('fasttext-model', type=click.Path(exists=True, dir_okay=False))
@click.argument('test-data', type=click.File('r'))
@click.option('-o', '--output-json', help='Output JSONL file', type=click.File('w'))
def predict(model, fasttext_model, test_data, output_json):
    """
    Apply trained message segmenter to predict lines of an email or newsgroup message.

    Arguments:
        MODEL: Trained HDF5 segmenter model
        FASTTEXT_MODEL: pre-trained FastText embedding
        TEST_DATA: test message dump as JSON
    """
    logger.info('Loading FastText model...')
    load_fasttext_model(fasttext_model)

    segmenter = models.load_model(model)
    to_stdout = output_json is None

    logger.info('Predicting {}...'.format(test_data.name))
    while True:
        # Do not load more than 1k lines at once
        pred_seq = MailLinesSequence(test_data, labeled=False, batch_size=256, max_lines=1000)
        if len(pred_seq) == 0:
            break

        predictions = segmenter.predict_generator(
            pred_seq, verbose=(not to_stdout), steps=(None if not to_stdout else 10))
        export_mail_annotation_spans(predictions, pred_seq, output_json, verbose=to_stdout)

        if output_json:
            output_json.flush()

    if output_json:
        output_json.close()


@main.command()
@click.argument('model', type=click.Path(exists=True, dir_okay=False))
@click.argument('fasttext-model', type=click.Path(exists=True, dir_okay=False))
@click.argument('eval-data', type=click.File('r'))
def evaluate(model, fasttext_model, eval_data):
    """
    Evaluate a trained message segmenter.

    Arguments:
        MODEL: Trained HDF5 segmenter model
        FASTTEXT_MODEL: pre-trained FastText embedding
        EVAL_DATA: test message dump as JSON
    """
    logger.info('Loading FastText model...')
    load_fasttext_model(fasttext_model)

    def paragraph_accuracy(y_true, y_pred):
        return metrics.binary_accuracy(*binarize_pred_tensors('paragraph', y_true, y_pred))

    def quotation_accuracy(y_true, y_pred):
        return metrics.binary_accuracy(*binarize_pred_tensors('quotation', y_true, y_pred))

    def raw_code_accuracy(y_true, y_pred):
        return metrics.binary_accuracy(*binarize_pred_tensors('raw_code', y_true, y_pred))

    segmenter = models.load_model(model)
    segmenter.compile(optimizer=segmenter.optimizer, loss=segmenter.loss,
                      metrics=['categorical_accuracy',
                               paragraph_accuracy, quotation_accuracy, raw_code_accuracy])

    logger.info('Evaluating {}...'.format(eval_data.name))
    batch_size = 256
    eval_seq = MailLinesSequence(eval_data, labeled=True, batch_size=batch_size)

    cls_sum = np.zeros(len(label_map))
    for i, (_, labels) in enumerate(tqdm(eval_seq, desc='Counting labels', unit='samples', leave=False)):
        cls_sum = np.add(cls_sum, np.sum(labels, axis=0))
        if i == len(eval_seq) - 1:
            break

    click.echo('Ground-truth class distribution:')
    cls_sum /= len(eval_seq) * batch_size
    cls_sum = sorted(enumerate(cls_sum), key=lambda x: x[1], reverse=True)
    for i, prob in cls_sum:
        click.echo(' {: >19}: {:.4f}'.format(label_map_inverse[i], prob))

    logger.info('Predicting samples...')
    segmenter.evaluate_generator(eval_seq, verbose=True)


class MailLinesSequence(Sequence):
    def __init__(self, input_file, labeled=True, batch_size=None, line_shape=(MAX_LEN, INPUT_DIM),
                 input_is_raw_text=False, max_lines=None):
        self.labeled = labeled
        self.mail_lines = []
        self.mail_start_index_map = {}
        self.mail_end_index_map = {}

        self.batch_size = batch_size
        self.line_shape = line_shape

        if self.labeled:
            self.padding_line = [(None, label_map_onehot['<pad>'])]
        else:
            self.padding_line = [None]

        if not input_is_raw_text:
            if type(input_file) is str:
                self._load_jsonl(open(input_file, 'r'), max_lines)
            else:
                self._load_jsonl(input_file, max_lines)
        else:
            self._load_raw_text(input_file, max_lines)

    def _load_jsonl(self, json_file, max_lines):
        context_padding = self.padding_line * CONTEXT

        for i, json_text in enumerate(json_file):
            mail_json = json.loads(json_text)

            lines = None
            if not self.labeled:
                lines = [l + '\n' for l in mail_json['text'].split('\n')]

            elif self.labeled and mail_json['labels']:
                lines = [l for l in label_lines(mail_json)]

            # Skip overly long mails (probably just excessive log data)
            if len(lines) > 5000:
                continue

            if lines:
                self.mail_start_index_map[len(self.mail_lines) + CONTEXT] = mail_json
                self.mail_end_index_map[len(self.mail_lines) + CONTEXT + len(lines)] = mail_json
                self.mail_lines.extend(context_padding + lines + context_padding)

            if max_lines is not None and i >= max_lines:
                break

        if self.batch_size is None:
            self.batch_size = len(self.mail_lines)

    def _load_raw_text(self, raw_text, max_lines):
        if max_lines is not None:
            lines = [l + '\n' for l in raw_text.split('\n')[:max_lines]]
        else:
            lines = [l + '\n' for l in raw_text.split('\n')]

        if lines:
            context_padding = self.padding_line * CONTEXT
            self.mail_lines.extend(context_padding + lines + context_padding)

        if self.batch_size is None:
            self.batch_size = len(self.mail_lines)

    def __len__(self):
        return int(np.ceil(len(self.mail_lines) / self.batch_size))

    def __getitem__(self, index):
        index = index * self.batch_size

        batch = np.empty((self.batch_size,) + self.line_shape)
        batch_prev = np.empty((self.batch_size,) + self.line_shape)
        batch_context = np.empty((self.batch_size, CONTEXT * 2 + 1) + self.line_shape)
        batch_labels = np.empty((self.batch_size, OUTPUT_DIM))

        end_index = index + self.batch_size if self.batch_size is not None else len(self.mail_lines)

        padding_lines = self.padding_line * CONTEXT
        mail_slice = padding_lines + self.mail_lines[index:end_index] + padding_lines

        for i, line in enumerate(mail_slice):
            if i < CONTEXT or i >= len(mail_slice) - CONTEXT:
                continue

            if self.labeled:
                batch_labels[i - CONTEXT] = line[1]
                # line_text = line[0] if line[0] is not None else '<PAD>\n'
                # click.echo('{:>20}    --->    {}'.format(label_map_inverse[np.argmax(line[1])], line_text), end='')

            line_vecs = []
            for c in chain(mail_slice[i - CONTEXT:i], [line], mail_slice[i + 1:i + 1 + CONTEXT]):
                if self.labeled:
                    c, _ = c    # type: tuple

                # Check if this is a padding line
                if c is None:
                    line_vecs.append(np.ones(self.line_shape) * -1)
                else:
                    line_vecs.append(pad_2d_sequence(get_word_vectors(c), self.line_shape[0]))

            batch[i - CONTEXT] = line_vecs[CONTEXT]
            batch_prev[i - CONTEXT] = line_vecs[CONTEXT - 1]
            batch_context[i - CONTEXT] = np.stack(line_vecs)

        if self.labeled:
            return [batch, batch_prev, batch_context], batch_labels

        return [batch, batch_prev, batch_context]


def binarize_pred_tensors(cls, *tensors):
    """
    Binarize multi-class prediction tensors to the given class.
    """
    cls = label_map[cls] if type(cls) is str else cls
    return [K.cast(K.equal(K.argmax(t, axis=-1), cls), t.dtype) for t in tensors]


def binarized_precision(y_true, y_pred, cls='paragraph'):
    """
    Accuracy metric measuring only binary accuracy between `cls` labels and 'the rest'.
    """
    y_true = K.equal(K.argmax(y_true, axis=-1), label_map[cls])
    y_pred = K.equal(K.argmax(y_pred, axis=-1), label_map[cls])
    return K.mean(K.equal(y_true, y_pred))


def pad_2d_sequence(seq, max_len):
    if seq.shape[0] > max_len:
        pivot_idx = int(np.ceil(max_len * .75))
        seq = np.concatenate((seq[:pivot_idx], seq[seq.shape[0] - max_len + pivot_idx:]))

    return np.pad(seq, ((0, max(0, max_len - seq.shape[0])), (0, 0)), 'constant')


def train_model(input_file, output_model, validation_input=None):
    tb_callback = callbacks.TensorBoard(log_dir='./data/graph/' + str(datetime.now()), update_freq=1000,
                                        histogram_freq=0, write_grads=True, write_graph=False, write_images=False)
    es_callback = callbacks.EarlyStopping(monitor='val_loss', verbose=1, patience=5)
    cp_callback = callbacks.ModelCheckpoint(output_model + '.epoch-{epoch:02d}.loss-{val_loss:.2f}.h5')

    def get_base_line_model():
        line_input = layers.Input(shape=(MAX_LEN, INPUT_DIM))
        masking = layers.Masking(0)(line_input)
        bi_seq = layers.Bidirectional(layers.GRU(128), merge_mode='sum')(masking)
        bi_seq = layers.BatchNormalization()(bi_seq)
        bi_seq = layers.Activation('relu')(bi_seq)
        return line_input, bi_seq

    def get_context_model():
        context_input = layers.Input(shape=(CONTEXT * 2 + 1, MAX_LEN, INPUT_DIM))
        conv2d = layers.Conv2D(128, (4, 4))(context_input)
        conv2d = layers.BatchNormalization()(conv2d)
        conv2d = layers.Activation('relu')(conv2d)
        conv2d = layers.Conv2D(64, (3, 3))(conv2d)
        conv2d = layers.Activation('relu')(conv2d)
        conv2d = layers.MaxPooling2D(2)(conv2d)
        flatten = layers.Flatten()(conv2d)
        dense = layers.Dense(128)(flatten)
        dense = layers.Activation('relu')(dense)
        return context_input, dense

    line_input_cur, line_model_cur = get_base_line_model()
    line_input_prev, line_model_prev = get_base_line_model()
    context_input, context_model = get_context_model()

    concat = layers.concatenate([line_model_cur, line_model_prev, context_model])
    dropout = layers.Dropout(0.25)(concat)
    dense_2 = layers.Dense(OUTPUT_DIM)(dropout)
    output = layers.Activation('softmax')(dense_2)

    segmenter = models.Model(inputs=[line_input_cur, line_input_prev, context_input], outputs=output)
    segmenter.compile(optimizer='adam', loss='categorical_hinge',
                      metrics=['categorical_accuracy'])
    segmenter.summary()

    train_seq = MailLinesSequence(input_file, labeled=True, batch_size=BATCH_SIZE)
    val_seq = MailLinesSequence(validation_input, labeled=True) if validation_input else None

    segmenter.fit_generator(train_seq, epochs=15, validation_data=val_seq, shuffle=True,
                             max_queue_size=100, callbacks=[tb_callback, es_callback, cp_callback])


def predict_raw_text(line_model, email):
    pred_seq = MailLinesSequence(email, labeled=False, input_is_raw_text=True)
    return (pred for i, pred in
            enumerate(post_process_labels(pred_seq.mail_lines, line_model.predict_generator(pred_seq)))
            if CONTEXT <= i < len(pred_seq.mail_lines) - CONTEXT)


def reformat_raw_text_recursive(line_model, email, exclude_classes=None, max_depth=10):
    """
    Predicts and recursively reformats an email.
    Nested quotations will be parsed, quotation markers and symbols are removed and
    the contents are predicted again.

    :param line_model: input model
    :param email: input email
    :param exclude_classes: exclude classes (default: signatures and technical)
    :param max_depth: maximum recursion depth
    :return: nested line predictions
    """

    if exclude_classes is None:
        exclude_classes = ['personal_signature', 'mua_signature', 'technical']

    def parse_quotation(lines):
        prefix = os.path.commonprefix([l for l in lines if l.lstrip().startswith('>') or l.lstrip().startswith('|')])
        prefix = prefix.replace('\n', '')
        text = ''
        for l in lines:
            text += re.sub(r'^\s*{}'.format(re.escape(prefix)), '', l.rstrip()).lstrip() + '\n'
        return text.strip() + '\n'

    def nest_quotation_markers(lines):
        nested = []
        for l in lines:
            if nested and type(l) is list and type(nested[-1]) is tuple and nested[-1][1] == 'quotation_marker':
                nested[-1] = [(re.sub(r'^[>|]+\s*', '', nested[-1][0].rstrip() + '\n'), nested[-1][1])] + l
            else:
                nested.append(l)
        return nested

    def strip_empty_boundaries(lines):
        stripped = []
        for i, l in enumerate(lines):
            if type(l) is tuple and l[1] == '<empty>' and i < len(lines) - 1:
                for l2 in lines[i + 1:]:
                    if type(l2) is tuple and l2[1] != '<empty>':
                        stripped.append(l)
                        break
            elif type(l) is not tuple or l[1] != '<empty>':
                stripped.append(l)
        return stripped

    def combine_lines(lines, classes_collapse_newline):
        combined = []
        for l in lines:
            if combined and type(l) is tuple and type(combined[-1]) is tuple and l[1] == combined[-1][1]:
                if l[0].strip() == '':
                    continue
                delim = ' ' if l[1] in classes_collapse_newline else '\n'
                combined[-1] = (combined[-1][0].rstrip() + delim + l[0], l[1])
            else:
                combined.append(l)
        return combined

    def recurse(text, depth=0):
        predictions = predict_raw_text(line_model, text)
        lines = []
        quotation_lines = []
        for line, cls in predictions:
            if cls in exclude_classes:
                continue

            if cls == 'quotation' and depth < max_depth:
                quotation_lines.append(line)
                continue

            if quotation_lines:
                quot = parse_quotation(quotation_lines)
                if quot.strip():
                    rec = recurse(quot, depth + 1)
                    if rec:
                        lines.append(rec)
                quotation_lines.clear()

            lines.append((line, cls))

        if quotation_lines and depth < max_depth:
            quot = parse_quotation(quotation_lines)
            if quot.strip():
                rec = recurse(quot, depth + 1)
                if rec:
                    lines.append(rec)

        lines = combine_lines(lines, ['quotation_marker', 'closing'])
        lines = strip_empty_boundaries(lines)
        lines = nest_quotation_markers(lines)
        lines = strip_empty_boundaries(lines)
        return lines

    return recurse(email)


def post_process_labels(lines, labels_softmax):
    lines = ([None] * CONTEXT) + lines + ([None] * CONTEXT)
    sm_pad = np.ones((CONTEXT, OUTPUT_DIM)) * -1
    labels_softmax = np.concatenate((sm_pad, labels_softmax, sm_pad))

    for i, (line, label) in enumerate(zip(lines, labels_softmax)):
        # Skip padding
        if i < CONTEXT:
            continue
        if i >= len(lines) - CONTEXT:
            break

        label_argmax = np.argmax(label)
        label_argsort = np.argsort(label)[::-1]
        label_text = label_map_inverse[label_argmax]

        context = min(3, CONTEXT)

        prev_l = [label_map_inverse[np.argmax(l)] for l in labels_softmax[i - context:i]]
        next_l = [label_map_inverse[np.argmax(l)] for l in labels_softmax[i + 1:i + 1 + context]]

        prev_set = set([l for l in prev_l if l not in ['<empty>', '<pad>']])
        next_set = set([l for l in next_l if l not in ['<empty>', '<pad>']])

        if line is None:
            yield '<PAD>\n', '<pad>'
            labels_softmax[i] = label_map_onehot['<pad>']
            continue

        # Correct <empty>
        if line.strip() == '':
            label_text = '<empty>'

        # Empty lines have to be empty
        elif (label_text == '<empty>' and line.strip() != '') or label_text == '<pad>':
            label_text = prev_l[-1] if prev_l[-1] not in ['<empty>', '<pad>'] else 'paragraph'

        # Bleeding quotations
        elif label_text == 'quotation' and prev_l[-1] == 'quotation' \
                and lines[i - 1].strip() and lines[i - 1].strip() \
                and next_l[0] != 'quotation' and lines[i - 1].strip()[0] != line.strip()[0] \
                and prev_l[-1] not in ['<empty>', '<pad>']:
            label_text = prev_l[-1]

        # Quotations
        elif label_text not in ['quotation', 'quotation_marker', 'inline_header'] \
                and (line.strip().startswith('>') or line.strip().startswith('|')) \
                and (label_map['quotation'] in label_argsort[:3] or prev_l[-1] == 'quotation'):
            label_text = 'quotation'

        # Quotation markers
        elif label_text == 'quotation' and prev_l[-1] in ['<empty>', '<pad>'] \
                and label_map['quotation_marker'] in label_argsort[:3]:
            label_text = 'quotation_marker'

        # Interrupted short blocks
        elif label_text != prev_l[-1] and next_l[0] == prev_l[-1] \
                and prev_l[-1] in ['closing', 'personal_signature', 'mua_signature', 'inline-header', 'technical']:
            label_text = prev_l[-1]

        # Interrupted long blocks
        elif len(prev_set) == 1 and label_text != [*prev_set][0] and [*prev_set][0] in next_set \
                and [*prev_set][0] in ['mua_signature', 'personal_signature',
                                       'patch', 'code', 'tabular', 'technical'] \
                and label_map[[*prev_set][0]] == label_argsort[1]:
            label_text = [*prev_set][0]

        # Interrupting stray classes
        elif label_text in ['technical', 'mua_signature', 'personal_signature', 'patch', 'tabular'] \
                and prev_l[-1] != label_text and prev_l[-1] not in ['<pad>', '<empty>'] \
                and (next_l[0] == prev_l[-1] or (next_l[1] == prev_l[-1] and next_l[0] == '<empty>')):
            label_text = prev_l[-1]

        labels_softmax[i] = label_map_onehot[label_text]
        yield line, label_text


def export_mail_annotation_spans(predictions_softmax, pred_sequence, output_file=None, verbose=True):
    text = ''
    annotations = []
    prev_label = None
    cur_label = '<pad>'
    start_offset = 0
    mail_dict = None
    skip_lines = CONTEXT

    def write_annotations(d, a):
        if not a or 'text' not in d or not d['text']:
            return

        d = {k: d[k] for k in d if k != 'id'}
        d.update({'labels': a, 'text': d['text'].lstrip()})

        json.dump(d, output_file)
        output_file.write('\n')

    for i, (line, label_text) in enumerate(post_process_labels(pred_sequence.mail_lines, predictions_softmax)):
        # Skip padding
        if i < skip_lines:
            continue
        skip_lines = i

        cur_label = label_text
        if prev_label is None:
            prev_label = cur_label

        if i in pred_sequence.mail_start_index_map:
            if verbose:
                click.echo(' {0:>>20}    --->    <<< MAIL START >>>'.format(''))
            mail_dict = pred_sequence.mail_start_index_map[i]

        cur_offset = len(text) - 1
        text += line
        text = text.lstrip()

        if i in pred_sequence.mail_end_index_map:
            if output_file:
                if prev_label not in ['<pad>', '<empty>']:
                    annotations.append((start_offset, cur_offset, prev_label))
                write_annotations(mail_dict, annotations)

            mail_dict = None
            annotations.clear()
            start_offset = 0
            prev_label = None
            text = ''
            skip_lines += CONTEXT * 2
            continue

        if verbose:
            click.echo(' {:>20}    --->    {}'.format(label_text, line), nl=False)

        if cur_label != prev_label:
            if output_file and prev_label not in ['<pad>', '<empty>']:
                annotations.append((start_offset, cur_offset, prev_label))

            start_offset = cur_offset + 1
            prev_label = cur_label

    if output_file and mail_dict:
        if cur_label not in ['<empty>', '<pad>']:
            annotations.append((start_offset, len(text) - 1, cur_label))
        write_annotations(mail_dict, annotations)


def get_annotations_from_dict(d):
    """
    Get annotations from Doccano either one of the two Doccano export formats.
    This is a little compatibility hack make both formats work.
    """
    if 'annotations' in d:
        return d['annotations']
    elif 'labels' in d:
        return [{'start_offset': a[0], 'end_offset': a[1], 'label': a[2]} for a in d['labels']]


def label_lines(doc):
    lines = [l + '\n' for l in doc['text'].split('\n')]
    annotations = sorted(get_annotations_from_dict(doc), key=lambda a: a['start_offset'], reverse=True)
    offset = 0
    for l in lines:
        end_offset = offset + len(l)

        if annotations and offset > annotations[-1]['end_offset']:
            annotations.pop()

        if not annotations or not l.strip():
            yield l, label_map_onehot['<empty>']
            offset = end_offset
            continue

        if offset < annotations[-1]['end_offset'] and end_offset > annotations[-1]['start_offset']:
            yield l, label_map_onehot[annotations[-1]['label']]
        else:
            yield l, label_map_onehot['<empty>']

        offset = end_offset


_model = None


def load_fasttext_model(model_path):
    global _model
    if not _model:
        _model = fastText.load_model(model_path)


def get_word_vectors(text):
    try:
        matrix = [_model.get_word_vector(w) for w in fastText.tokenize(util.normalize_message_text(text))]
    except Exception as e:
        logger.error('Failed to tokenize line: {}'.format(e))
        matrix = [get_word_vector('')]

    return np.array(matrix)


def get_word_vector(word):
    if _model is None:
        raise RuntimeError("FastText vectors not loaded. Call load_fasttext_model() first.")

    try:
        return _model.get_word_vector(word)
    except Exception as e:
        logger.error('Failed to obtain word vector: {}'.format(e))
        return _model.get_word_vector('')


if __name__ == '__main__':
    main()
