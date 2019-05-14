#!/usr/bin/env python3

from datetime import datetime
import fastText
from keras import callbacks, layers, models
import math
import numpy as np
import json
import plac
import re
import sys


label_map_int = {
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


def labels_to_onehot(labels_dict):
    onehots = np.eye(len(labels_dict))
    onehot_dict = {l: onehots[i] for i, l in enumerate(labels_dict)}
    return onehot_dict


label_map_inverse = {label_map_int[k]: k for k in label_map_int}
label_map = labels_to_onehot(label_map_int)

INPUT_DIM = 100
OUTPUT_DIM = len(label_map)
BATCH_SIZE = 128
MAX_LEN = 15
CONTEXT = 3


@plac.annotations(
    cmd=('Command', 'positional', None, str, None, 'CMD'),
    input_file=('Input JSONL file', 'positional', None, str, None, 'JSONL'),
    model=('Keras model', 'positional', None, str, None, 'K5'),
    fasttext_model=('FastText Model', 'positional', None, str, None, 'FASTTEXT_BIN'),
    output_json=('Output JSONL file', 'option', 'o', str, None, 'OUTPUT')
)
def main(cmd, input_file, model, fasttext_model, output_json=None):
    labeled_mails = []
    unlabeled_mails = []
    for line in open(input_file).readlines():
        mail_json = json.loads(line)
        if not mail_json['annotations']:
            unlabeled_mails.append(([l + '\n' for l in mail_json['text'].split('\n')], mail_json))
            continue

        labeled_mails.append([l for l in label_lines(mail_json)])

    print('Labeled emails:', len(labeled_mails))
    print('Unlabeled emails:', len(unlabeled_mails))

    print('Loading FastText model...')
    load_fasttext_model(fasttext_model)

    if cmd == 'train':
        train_model(labeled_mails, model)
    elif cmd == 'predict':
        predict(unlabeled_mails, model, output_json)
    else:
        print('Invalid command.', file=sys.stderr)
        exit(1)


def train_model(labeled_mails, output_model):
    lines_matrix = []
    labels = []
    for mail in labeled_mails:
        pad_rows(lines_matrix, labels, CONTEXT)

        for line, label in mail:
            #print('{:>20}    train --->    {}'.format(label_map_inverse[np.argmax(label)], line), end='')
            lines_matrix.append(pad_2d_sequence(get_word_vectors(line), MAX_LEN))
            labels.append(label)

        pad_rows(lines_matrix, labels, CONTEXT)

    lines_matrix = contextualize(lines_matrix, CONTEXT)
    labels = np.array(labels)

    tb_callback = callbacks.TensorBoard(log_dir='./data/graph/' + str(datetime.now()), update_freq=1000,
                                        histogram_freq=0, write_grads=True, write_graph=False, write_images=False)
    es_callback = callbacks.EarlyStopping(monitor='val_loss', verbose=1)

    # Line model
    line_model = models.Sequential()
    line_model.add(layers.Conv1D(5, 3, input_shape=(None, INPUT_DIM)))
    line_model.add(layers.MaxPooling1D(3))
    line_model.add(layers.Activation('selu'))
    line_model.add(layers.Bidirectional(layers.GRU(128), merge_mode='sum'))
    line_model.add(layers.BatchNormalization())
    line_model.add(layers.Activation('selu'))
    line_model.add(layers.Dropout(0.5))
    line_model.add(layers.Dense(OUTPUT_DIM))
    line_model.add(layers.Activation('softmax'))
    line_model.compile(optimizer='adam', loss='categorical_crossentropy',
                       metrics=['categorical_accuracy', 'mean_squared_error'])
    line_model.summary()

    line_model.fit(lines_matrix, np.array(labels), validation_split=0.1, epochs=15, batch_size=BATCH_SIZE,
                   callbacks=[tb_callback, es_callback])
    line_model.save(output_model + '.hdf5')


def predict(unlabeled_mails, input_model, output_json=None):
    line_model = models.load_model(input_model + '.hdf5')

    output_json_file = None
    if output_json:
        output_json_file = open(output_json, 'w')

    for mail_lines, mail_dict in unlabeled_mails:
        if len(mail_lines) > 10000:
            continue

        lines_matrix = []
        pad_rows(lines_matrix, [], CONTEXT)
        lines_matrix.extend([pad_2d_sequence(get_word_vectors(l), MAX_LEN) for l in mail_lines])
        pad_rows(lines_matrix, [], CONTEXT)

        lines_matrix = contextualize(lines_matrix, CONTEXT)
        predictions = line_model.predict(lines_matrix)

        mail_lines = (['<PAD>\n'] * CONTEXT) + mail_lines + (['<PAD>\n'] * CONTEXT)
        export_mail_annotation_spans(mail_lines, mail_dict, predictions, output_json_file)

    if output_json_file:
        output_json_file.close()


def post_process_labels(lines, labels_softmax):
    for i, (line, label) in enumerate(zip(lines, labels_softmax)):
        # Skip padding
        if i < CONTEXT:
            continue
        if i >= len(lines) - CONTEXT:
            break

        label_argmax = np.argmax(label)
        label_argsort = np.argsort(label)
        label_text = label_map_inverse[label_argmax]

        prev_l = [label_map_inverse[np.argmax(l)] for l in labels_softmax[i - CONTEXT:i]]
        next_l = [label_map_inverse[np.argmax(l)] for l in labels_softmax[i + 1:i + 1 + CONTEXT]]

        prev_set = set([l for l in prev_l if l not in ['<empty>', '<pad>']])
        next_set = set([l for l in next_l if l not in ['<empty>', '<pad>']])

        # Correct <empty>
        if line.strip() == '':
            label_text = '<empty>'

        # Empty lines have to be empty
        elif (label_text == '<empty>' and line.strip() != '') or label_text == '<pad>':
            label_text = prev_l[-1] if prev_l[-1] != '<empty>' else 'paragraph'

        # Bleeding quotations
        elif label_text == 'quotation' and prev_l[-1] == 'quotation' \
                and lines[i - 1].strip() and lines[i - 1].strip() \
                and next_l[0] != 'quotation' and lines[i - 1].strip()[0] != line.strip()[0]:
            label_text = prev_l[-1]

        # Quotation markers
        elif label_text == 'quotation' and prev_l[-1] in ['<empty>', '<pad>'] \
                and label_map_int['quotation_marker'] in label_argsort[:3]:
            label_text = 'quotation_marker'

        # Interrupted closings / signatures
        elif label_text != prev_l[-1] and next_l[0] == prev_l[-1] \
                and prev_l[-1] in ['closing', 'personal_signature', 'mua_signature']:
            label_text = prev_l[-1]

        # Personal signatures in MUA signatures
        elif label_text == 'personal_signature' and prev_l[-1] == 'mua_signature' \
                and (next_l[0] == 'mua_signature' or next_l[0] == '<pad>'):
            label_text = 'mua_signature'

        # MUA signatures in personal signatures
        elif label_text == 'mua_signature' and prev_l[-1] == 'personal_signature' \
                and (next_l[0] == 'personal_signature' or next_l[0] == '<pad>'):
            label_text = 'personal_signature'

        # Interrupted blocks
        elif len(prev_set) == 1 and label_text != [*prev_set][0] and [*prev_set][0] in next_set \
                and [*prev_set][0] in ['mua_signature', 'personal_signature', 'patch', 'code', 'tabular']:
            label_text = [*prev_set][0]

        # Stray technical
        elif label_text == 'technical' and prev_l[-1] not in ['technical', '<empty>']:
            label_text = prev_l[-1]

        labels_softmax[i] = label_map[label_text]
        yield line, label_text


def export_mail_annotation_spans(mail_lines, mail_dict, predictions_softmax, output_file=None):
    text = ''
    annotations = []
    prev_label = None
    cur_label = '<pad>'
    start_offset = 0

    for i, (line, label_text) in enumerate(post_process_labels(mail_lines, predictions_softmax)):
        cur_label = label_map_inverse[np.argmax(predictions_softmax[i])]
        if prev_label is None:
            prev_label = cur_label

        cur_offset = len(text) - 1
        text = text + mail_lines[i]
        if cur_label != prev_label:
            if output_file and prev_label not in ['<pad>', '<empty>']:
                annotations.append((start_offset, cur_offset, prev_label))

            start_offset = cur_offset + 1
            prev_label = cur_label

        print('{:>20}    --->    {}'.format(label_text, line), end='')

    print()

    if not output_file:
        return

    if cur_label not in ['<empty>', '<pad>']:
        annotations.append((start_offset, len(text) - 1, cur_label))

    d = mail_dict.copy()

    if 'id' in d:
        del d['id']

    d.update({'labels': annotations, 'annotations': annotations})
    json.dump(d, output_file)
    output_file.write('\n')


def contextualize(lines, context=CONTEXT):
    lines_copy = []
    pad_rows(lines_copy, [], context)
    lines_copy.extend(lines)
    pad_rows(lines_copy, [], context)

    c_lines = []
    for i, line in enumerate(lines_copy):
        if i < context or i >= len(lines_copy) - context:
            continue

        prev_vec = lines_copy[i - context:i]
        next_vec = lines_copy[i + 1:i + 1 + context]

        c_lines.append(np.concatenate(prev_vec + [line] + next_vec))

    return np.array(c_lines)


def pad_rows(rows, labels, pad=1, shape=(MAX_LEN, INPUT_DIM)):
    for _ in range(pad):
        rows.append(np.ones(shape) * -1)
        labels.append(label_map['<pad>'])


def pad_2d_sequence(seq, max_len):
    return np.pad(seq[:max_len], ((0, max(0, max_len - seq.shape[0])), (0, 0)), 'constant')


def label_lines(doc):
    lines = [l + '\n' for l in doc['text'].split('\n')]
    annotations = sorted(doc['annotations'], key=lambda a: a['start_offset'], reverse=True)
    offset = 0
    for l in lines:
        end_offset = offset + len(l)

        if annotations and offset > annotations[-1]['end_offset']:
            annotations.pop()

        if not annotations or not l.strip():
            yield l, label_map['<empty>']
            offset = end_offset
            continue

        if offset < annotations[-1]['end_offset'] and end_offset > annotations[-1]['start_offset']:
            yield l,  label_map[annotations[-1]['label']]
        else:
            yield l, label_map['<empty>']

        offset = end_offset


_model = None


def load_fasttext_model(model_path):
    global _model
    _model = fastText.load_model(model_path)


def get_word_vectors(text):
    text = re.sub(r'([a-zA-Z0-9_\-\./+]+)@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.)|' +
                  r'(([a-zA-Z0-9\-]+\.)+))([a-zA-Z]{2,4}|[0-9]{1,3})(\]?)', 'mail@address', text)
    matrix = [_model.get_word_vector(w) for w in fastText.tokenize(text)]
    start_line = np.ones(INPUT_DIM)
    return np.array([start_line] + matrix)


def get_word_vector(word):
    return _model.get_word_vector(word)


if __name__ == '__main__':
    plac.call(main)
