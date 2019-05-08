#!/usr/bin/env python3

import fastText
from keras import callbacks, layers, models
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from keras_self_attention import SeqSelfAttention
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
MAX_LEN = 15
CONTEXT = 3


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

    # Line model
    # lines_matrix = np.array(lines_matrix)
    lines_matrix = np.array(contextualize(lines_matrix, CONTEXT))
    labels = np.array(labels)

    tb_callback = callbacks.TensorBoard(log_dir='./data/graph', update_freq=1000, histogram_freq=0,
                                        write_grads=True, write_graph=False, write_images=False)

    # deep_model = models.load_model(output_model + '.k5', custom_objects={'SeqSelfAttention': SeqSelfAttention})
    deep_model = models.Sequential()
    deep_model.add(layers.Masking(0.0, input_shape=(None, INPUT_DIM)))
    deep_model.add(layers.Bidirectional(layers.GRU(100, return_sequences=True, activation='selu'), merge_mode='sum'))
    deep_model.add(SeqSelfAttention(attention_activation='selu'))
    deep_model.add(layers.Bidirectional(layers.GRU(50, activation='selu'), merge_mode='sum'))
    deep_model.add(layers.Dropout(0.2))
    deep_model.add(layers.Dense(OUTPUT_DIM, activation='softmax'))
    deep_model.compile(optimizer='adam', loss='categorical_crossentropy',
                       metrics=['categorical_accuracy', 'mean_squared_error'])
    deep_model.summary()
    deep_model.fit(lines_matrix, np.array(labels), validation_split=0.1, epochs=100, batch_size=BATCH_SIZE,
                   callbacks=[tb_callback])

    deep_model.save(output_model + '.k5')

    # Sequence model
    mail_size = 20
    split = len(lines_matrix) // mail_size
    pad_len = split - (len(lines_matrix) % split)
    if pad_len > 0:
        lines_matrix = np.concatenate((lines_matrix, np.zeros((pad_len,) + lines_matrix.shape[1:])))
        labels = np.concatenate((labels, np.zeros((pad_len,) + labels.shape[1:])))

    labels_split = np.stack(np.split(labels, split))
    pred = deep_model.predict(lines_matrix, verbose=1)
    pred_split = np.stack(np.split(pred, split))

    crf_model = models.Sequential()
    crf_model.add(layers.Masking(0.0, input_shape=(None, OUTPUT_DIM)))
    crf_model.add(CRF(OUTPUT_DIM))
    crf_model.compile(optimizer='adam', loss=crf_loss, metrics=[crf_viterbi_accuracy])
    crf_model.summary()
    crf_model.fit(pred_split, labels_split, batch_size=50, epochs=200)

    crf_model.save(output_model + '_crf.k5')


def predict(unlabeled_mails, input_model):
    custom_objects = {'CRF': CRF,
                      'crf_loss': crf_loss,
                      'crf_viterbi_accuracy': crf_viterbi_accuracy,
                      'SeqSelfAttention': SeqSelfAttention}

    deep_model = models.load_model(input_model + '.k5', custom_objects=custom_objects)
    crf_model = models.load_model(input_model + '_crf.k5', custom_objects=custom_objects)

    for mail in unlabeled_mails:
        lines_matrix = []
        pad_rows(lines_matrix, [], CONTEXT)
        lines_matrix.extend([pad_2d_sequence(get_word_vectors(l), MAX_LEN) for l in mail])
        pad_rows(lines_matrix, [], CONTEXT)

        # lines_matrix = np.array(lines_matrix)
        lines_matrix = np.array(contextualize(lines_matrix, CONTEXT))

        predictions_intermediate = deep_model.predict(lines_matrix)
        predictions_intermediate = np.reshape(predictions_intermediate, (1,) + predictions_intermediate.shape)

        predictions = crf_model.predict(predictions_intermediate)
        predictions = np.argmax(np.reshape(predictions, (predictions.shape[1:])), axis=1)
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


def pad_rows(rows, labels, pad=1, shape=(MAX_LEN, INPUT_DIM)):
    for _ in range(pad):
        rows.append(np.zeros(shape))
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
