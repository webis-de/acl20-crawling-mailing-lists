#!/usr/bin/env python3

import fastText
from keras import layers, models
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
import plac
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
    'stacktrace': 12,
    'tabular': 13,
    'technical': 14,
    'visual_separator': 15
}


def labels_to_onehot(labels_dict):
    onehots = np.eye(len(labels_dict))
    onehot_dict = {l: onehots[i] for i, l in enumerate(labels_dict)}
    onehot_dict[None] = np.zeros(len(labels_dict))
    return onehot_dict


label_map_inverse = {label_map_int[k]: k for k in label_map_int}
label_map = labels_to_onehot(label_map_int)

INPUT_DIM = 100
OUTPUT_DIM = len(label_map) - 1
BATCH_SIZE = 50
MAX_LEN = 25
CONTEXT = 2


@plac.annotations(
    cmd=('Command', 'positional', None, str, None, 'CMD'),
    input_file=('Input JSONL file', 'positional', None, str, None, 'FILE'),
    model=('Keras model', 'positional', None, str, None, 'FILE'),
    fasttext_model=('FastText Model', 'positional', None, str, None, 'FILE')
)
def main(cmd, input_file, model, fasttext_model):
    labeled_mails = []
    unlabeled_mails = []
    for line in open(input_file).readlines():
        mail_json = json.loads(line)
        if not mail_json['annotations']:
            unlabeled_mails.append([l + '\n' for l in mail_json['text'].split('\n')])
            continue

        labeled_mails.append([l for l in label_lines(mail_json)])

    print('Labeled emails:', len(labeled_mails))
    print('Unlabeled emails:', len(unlabeled_mails))

    print('Loading FastText model...')
    load_fasttext_model(fasttext_model)

    if cmd == 'train':
        train_model(labeled_mails, model)
    elif cmd == 'predict':
        predict(unlabeled_mails, model)
    else:
        print('Invalid command.', file=sys.stderr)
        exit(1)


def train_model(labeled_mails, output_model):
    lines_matrix = []
    mail_boundaries = []
    labels = []
    i = 0
    for mail in labeled_mails:
        pad_rows(lines_matrix, labels, CONTEXT)
        num_samples = 1

        for line, label in mail:
            lines_matrix.append(pad_2d_sequence(get_word_vectors(line), MAX_LEN))
            labels.append(label)
            num_samples += 1

        pad_rows(lines_matrix, labels, CONTEXT)
        num_samples += 1

        mail_boundaries.append((i, i + num_samples))
        i += num_samples

    # lines_matrix = np.array(lines_matrix)
    lines_matrix = np.array(contextualize(lines_matrix, CONTEXT))

    # model = models.load_model(output_model1)
    model = models.Sequential()
    model.add(layers.Masking(0.0, input_shape=(lines_matrix[0].shape[0], INPUT_DIM)))
    model.add(layers.Bidirectional(layers.GRU(100), merge_mode='sum'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(OUTPUT_DIM, activation='softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='rmsprop')
    model.summary()
    model.fit(lines_matrix, np.array(labels), validation_split=0.1, epochs=10, batch_size=BATCH_SIZE)

    model.save(output_model + '.k5')


def predict(unlabeled_mails, input_model):
    model = models.load_model(input_model + '.k5')

    for mail in unlabeled_mails:
        lines_matrix = []
        pad_rows(lines_matrix, [], CONTEXT)
        lines_matrix.extend([pad_2d_sequence(get_word_vectors(l), MAX_LEN) for l in mail])
        pad_rows(lines_matrix, [], CONTEXT)

        # lines_matrix = np.array(lines_matrix)
        lines_matrix = np.array(contextualize(lines_matrix, CONTEXT))

        predictions = np.argmax(model.predict(lines_matrix, batch_size=BATCH_SIZE), axis=1)
        for i, _ in enumerate(mail):
            p = predictions[i + CONTEXT]
            print('{:>20}    --->    {}'.format(label_map_inverse[p], mail[i]), end='')
        print()


def contextualize(lines, context=1):
    lines_copy = []
    pad_rows(lines_copy, [], context)
    lines_copy.extend(lines)
    pad_rows(lines_copy, [], context)

    c_lines = []
    for i, line in enumerate(lines_copy):
        if i < context or i >= len(lines_copy) - context:
            continue

        prev_vec = lines_copy[i - context:i]
        next_vec = lines_copy[i + 1:i + context + 1]

        c_lines.append(np.concatenate(prev_vec + [line] + next_vec))

    return c_lines


def pad_rows(rows, labels, pad=1):
    for _ in range(pad):
        rows.append(np.zeros((MAX_LEN, INPUT_DIM)))
        labels.append(label_map[None])


def pad_2d_sequence(seq, max_len):
    return np.pad(seq[:max_len], ((0, max(0, max_len - seq.shape[0])), (0, 0)), 'constant')


def label_lines(doc):
    lines = [l + '\n' for l in doc['text'].split('\n')]
    annotations = sorted(doc['annotations'], key=lambda a: a['start_offset'], reverse=True)
    offset = 0
    for l in lines:
        end_offset = offset + len(l) + 1

        if annotations and offset > annotations[-1]['end_offset']:
            annotations.pop()

        # skip annotations which span less than half the line
        if annotations and annotations[-1]['start_offset'] >= offset and \
                annotations[-1]['end_offset'] - annotations[-1]['start_offset'] < (end_offset - offset) / 2:
            annotations.pop()

        if not annotations:
            yield l, label_map['paragraph']
            continue

        if offset <= annotations[-1]['end_offset'] and end_offset >= annotations[-1]['start_offset']:
            yield l, label_map[annotations[-1]['label']]
        else:
            yield l, label_map['paragraph']

        offset = end_offset


_model = None


def load_fasttext_model(model_path):
    global _model
    _model = fastText.load_model(model_path)


def get_word_vectors(text):
    matrix = []
    for w in fastText.tokenize(text):
        matrix.append(_model.get_word_vector(w))

    return np.array(matrix)


def get_word_vector(word):
    return _model.get_word_vector(word)


if __name__ == '__main__':
    plac.call(main)
