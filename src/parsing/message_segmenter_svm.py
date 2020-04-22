#!/usr/bin/env python3
#
# This file contains legacy code that is only meant as a reproduction study
# of the 2005 paper "Email Data Cleaning" by Tang et al.
# For better message segmentation, use the deep segmenter from message_segmenter.py.

import json
import os
import pickle
import re

import click
import numpy as np
from sklearn import svm
from tqdm import tqdm
import tensorflow as tf

from util.mail_classification import get_annotations_from_dict

CONTENT = 0
HEADER = 1
SIGNATURE = 2
CODE = 3
QUOTATION = 4
EMPTY = 5

label_map = {
    'paragraph': CONTENT,
    'closing': CONTENT,
    'inline_headers': HEADER,
    'log_data': CONTENT,
    'mua_signature': SIGNATURE,
    'patch': CODE,
    'personal_signature': SIGNATURE,
    'quotation': QUOTATION,
    'quotation_marker': HEADER,
    'raw_code': CODE,
    'salutation': CONTENT,
    'section_heading': CONTENT,
    'tabular': CONTENT,
    'technical': CODE,
    'visual_separator': SIGNATURE,
    '<empty>': EMPTY
}

label_map_inverse = {
    CONTENT: 'content',
    HEADER: 'header',
    SIGNATURE: 'signature',
    CODE: 'code',
    QUOTATION: 'quotation',
    EMPTY: '<empty>'
}


@click.group()
def main():
    pass


@main.command()
@click.argument('train-data', type=click.Path(exists=True, dir_okay=False))
@click.argument('model-dir', type=click.Path(exists=True, file_okay=False))
def train(train_data, model_dir):
    labeled_mails, _ = load_mails(train_data)
    train_clf(labeled_mails, model_dir)


@main.command()
@click.argument('test-data', type=click.Path(exists=True, dir_okay=False))
@click.argument('model-dir', type=click.Path(exists=True, file_okay=False))
def predict(test_data, model_dir):
    _, unlabeled_mails = load_mails(test_data)
    list(tqdm(predict_mails(unlabeled_mails, model_dir), desc='Predicting lines'))


@main.command()
@click.argument('eval-data', type=click.Path(exists=True, dir_okay=False))
@click.argument('model-dir', type=click.Path(exists=True, file_okay=False))
@click.option('-v', '--verbose', is_flag=True, help='Show line predictions on STDOUT')
def evaluate(eval_data, model_dir, verbose):
    ground_truth, _ = load_mails(eval_data)
    pred_gen = predict_mails(ground_truth, model_dir, verbose)

    if not verbose:
        pred_gen = tqdm(pred_gen, desc='Predicting lines', leave=False)

    y_pred = np.array([(p[-2], p[-1]) for p in pred_gen])
    y_pred, y_true = [np.reshape(a, (a.shape[0],)) for a in np.split(y_pred, 2, axis=1)]

    click.echo('\nAccuracy: {:.4f}'.format(np.average(y_pred == y_true)))

    click.echo('Binarized Quotation Accuracy: {:.4f}'.format(
        np.average(np.where(y_pred == QUOTATION, 1, 0) == np.where(y_true == QUOTATION, 1, 0))))
    click.echo('Binarized Content Accuracy: {:.4f}'.format(
        np.average(np.where(y_pred == CONTENT, 1, 0) == np.where(y_true == CONTENT, 1, 0))))
    click.echo('Binarized Header Accuracy: {:.4f}'.format(
        np.average(np.where(y_pred == HEADER, 1, 0) == np.where(y_true == HEADER, 1, 0))))
    click.echo('Binarized Signature Accuracy: {:.4f}'.format(
        np.average(np.where(y_pred == SIGNATURE, 1, 0) == np.where(y_true == SIGNATURE, 1, 0))))
    click.echo('Binarized Code Accuracy: {:.4f}'.format(
        np.average(np.where(y_pred == CODE, 1, 0) == np.where(y_true == CODE, 1, 0))))

    confusion_mat = tf.math.confusion_matrix(labels=y_true, predictions=y_pred,
                                             num_classes=len(label_map_inverse)).numpy()
    click.echo('\nConfusion matrix:')
    with np.printoptions(precision=3, suppress=True, linewidth=9999, threshold=9999):
        click.echo(confusion_mat / np.sum(confusion_mat, axis=1)[:, np.newaxis])
    click.echo('Labels: ')
    click.echo([l for l in label_map_inverse.values()])


def load_mails(input_file):
    labeled_mails = []
    unlabeled_mails = []
    for line in open(input_file).readlines():
        mail_json = json.loads(line)
        if not mail_json.get('annotations') and not mail_json.get('labels'):
            unlabeled_mails.append([l + '\n' for l in mail_json['text'].split('\n')])
            continue

        labeled_mails.append(list(label_lines(mail_json)))

    click.echo('Labeled: {}'.format(len(labeled_mails)), err=True)
    click.echo('Unlabeled: {}'.format(len(unlabeled_mails)), err=True)

    return labeled_mails, unlabeled_mails


def train_clf(labeled_mails, model_dir):
    X_header, y_header = concat_vectors(labeled_mails, vectorize_line_header)
    X_signature, y_signature = concat_vectors(labeled_mails, vectorize_line_signature)
    X_code, y_code = concat_vectors(labeled_mails, vectorize_line_code)

    y_header_start, y_header_end = get_boundary_labels(y_header, HEADER)
    y_signature_start, y_signature_end = get_boundary_labels(y_signature, SIGNATURE)
    y_code_start, y_code_end = get_boundary_labels(y_code, CODE)

    click.echo('Training header start model', err=True)
    clf_header_start = svm.SVC(kernel='poly', gamma='scale')
    clf_header_start.fit(X_header, y_header_start)
    pickle.dump(clf_header_start, open(os.path.join(model_dir, 'header_start.model'), 'wb'))

    click.echo('Training header end model', err=True)
    clf_header_end = svm.SVC(kernel='poly', gamma='scale')
    clf_header_end.fit(X_header, y_header_end)
    pickle.dump(clf_header_end, open(os.path.join(model_dir, 'header_end.model'), 'wb'))

    click.echo('Training signature start model', err=True)
    clf_signature_start = svm.SVC(kernel='poly', gamma='scale')
    clf_signature_start.fit(X_signature, y_signature_start)
    pickle.dump(clf_signature_start, open(os.path.join(model_dir, 'signature_start.model'), 'wb'))

    click.echo('Training signature end model', err=True)
    clf_signature_end = svm.SVC(kernel='poly', gamma='scale')
    clf_signature_end.fit(X_signature, y_signature_end)
    pickle.dump(clf_signature_end, open(os.path.join(model_dir, 'signature_end.model'), 'wb'))

    click.echo('Training code start model', err=True)
    clf_code_start = svm.SVC(kernel='poly', gamma='scale')
    clf_code_start.fit(X_code, y_code_start)
    pickle.dump(clf_code_start, open(os.path.join(model_dir, 'code_start.model'), 'wb'))

    click.echo('Training code end model', err=True)
    clf_code_end = svm.SVC(kernel='poly', gamma='scale')
    clf_code_end.fit(X_code, y_code_end)
    pickle.dump(clf_code_end, open(os.path.join(model_dir, 'code_end.model'), 'wb'))


def predict_mails(mails, model_dir, verbose=True):
    clf_header_start = pickle.load(open(os.path.join(model_dir, 'header_start.model'), 'rb'))
    clf_header_end = pickle.load(open(os.path.join(model_dir, 'header_end.model'), 'rb'))
    clf_signature_start = pickle.load(open(os.path.join(model_dir, 'signature_start.model'), 'rb'))
    clf_signature_end = pickle.load(open(os.path.join(model_dir, 'signature_end.model'), 'rb'))
    clf_code_start = pickle.load(open(os.path.join(model_dir, 'code_start.model'), 'rb'))
    clf_code_end = pickle.load(open(os.path.join(model_dir, 'code_end.model'), 'rb'))

    for mail in mails:
        X_header, _ = vectorize_lines(mail, vectorize_line_header)
        X_signature, _ = vectorize_lines(mail, vectorize_line_signature)
        X_code, _ = vectorize_lines(mail, vectorize_line_code)

        y_header_start = clf_header_start.predict(X_header)
        y_header_end = clf_header_end.predict(X_header)
        y_signature_start = clf_signature_start.predict(X_signature)
        y_signature_end = clf_signature_end.predict(X_signature)
        y_code_start = clf_code_start.predict(X_code)
        y_code_end = clf_code_end.predict(X_code)

        in_header = False
        in_signature = False
        in_code = False
        ground_truth = None
        truth_exists = False
        for i, l in enumerate(mail):
            if type(l) is tuple:
                truth_exists = True
                l, ground_truth = l

            if y_signature_start[i] == 1:
                in_signature = True
                in_header = False
                in_code = False
            elif y_code_start[i] == 1:
                in_code = True
                in_header = False
                in_signature = False
            elif y_header_start[i] == 1:
                in_header = True
                in_signature = False
                in_code = False

            cls = CONTENT
            if in_header:
                cls = HEADER
            if in_signature:
                cls = SIGNATURE
            if in_code:
                cls = CODE
            if l.lstrip().startswith('>') or l.lstrip().startswith('|'):
                cls = QUOTATION
            if not l.strip():
                cls = EMPTY

            if y_header_end[i] == 1:
                in_header = False
            if y_signature_end[i] == 1:
                in_signature = False
            if y_code_end[i] == 1:
                in_code = False

            if truth_exists:
                if verbose:
                    click.echo('[ {: >17} ] {}'.format('{} ({})'.format(label_map_inverse[cls],
                                                                        cls == ground_truth), l))
                yield l, cls, ground_truth
            else:
                if verbose:
                    click.echo('[ {: >12} ] {}'.format(label_map_inverse[cls], l))
                yield l, cls


def concat_vectors(labeled_mails, callback):
    X = None
    y = None

    for mail in labeled_mails:
        X_temp, y_temp = vectorize_lines(mail, callback)
        if X is None:
            X = X_temp
            y = y_temp
        else:
            X = np.concatenate((X, X_temp))
            y = np.concatenate((y, y_temp))

    return X, y


def get_boundary_labels(y, filter_label):
    labels_start = np.zeros(len(y))
    labels_end = np.zeros(len(y))
    in_block = False
    for i, l in enumerate(y):
        if i > 0 and in_block and l != filter_label:
            for j in range(i, len(y)):
                if (y[j] != filter_label and y[j] != EMPTY) or j == len(y) - 1:
                    labels_end[i - 1] = 1
                    in_block = False
                    break

        if not in_block and l == filter_label:
            labels_start[i] = 1
            in_block = True

    if in_block:
        labels_end[-1] = 1

    return labels_start, labels_end


def label_lines(doc):
    lines = [l + '\n' for l in doc['text'].split('\n')]
    annotations = sorted(get_annotations_from_dict(doc), key=lambda a: a['start_offset'], reverse=True)
    offset = 0
    for l in lines:
        end_offset = offset + len(l)

        if annotations and offset > annotations[-1]['end_offset']:
            annotations.pop()

        l = l.rstrip()

        if not annotations:
            yield l, CONTENT if l else EMPTY
            offset = end_offset
            continue

        if offset < annotations[-1]['end_offset'] and end_offset > annotations[-1]['start_offset']:
            yield l, label_map[annotations[-1]['label']] if l else EMPTY
        else:
            yield l, CONTENT if l else EMPTY

        offset = end_offset


def vectorize_lines(lines, callback):
    vectors = []
    empty_lines = 0.0
    labels = np.zeros(len(lines))

    for i, l_tup in enumerate(lines):
        if type(l_tup) is tuple:
            l, labels[i] = l_tup
        else:
            l = l_tup

        if not l.strip():
            empty_lines += 1.0
        else:
            empty_lines = 0.0

        first_line = float(i == 0)
        last_line = float(i == len(lines) - 1)

        vectors.append((empty_lines, first_line, last_line) + tuple(map(float, callback(l))))

    matrix = np.zeros((len(lines), len(vectors[0]) * 3))
    for i, v in enumerate(vectors):
        prev_line = vectors[i - 1] if i != 0 else (-1,) * len(v)
        next_line = vectors[i + 1] if i != len(vectors) - 1 else (-1,) * len(v)
        matrix[i] = prev_line + v + next_line

    return matrix, labels


# noinspection DuplicatedCode
def vectorize_line_header(line):
    line_lower = line.lower()
    pos_word_start = re.search(r'^(?:from:|re:|aw:|fwd:|in article|in message)', line_lower) is not None
    pos_word_contains = re.search(r'(?:original message|fwd:|in reply to)', line_lower) is not None
    pos_word_end = re.search(r'(?:wrote:|said:|writes|commented:)$', line_lower) is not None
    negative_words = re.search(r'(?:hi|hello|hey|dear|regards|thanks|thank you|sincerely|good luck|cheers)',
                               line_lower) is not None
    email = re.search(r'^([a-zA-Z0-9_\-\.]+)@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.)|(([a-zA-Z0-9\-]+\.)+))' +
                      r'([a-zA-Z]{2,4}|[0-9]{1,3})(\]?)$', line_lower) is not None
    num_words = len(re.split(r'\s+', line))
    end_character = {':': 1, ';': 2, '"': 3, "'": 3, '?': 4, '!': 5, '.': 6}.get(line[-1], 0) if line else 0

    return pos_word_start, pos_word_contains, pos_word_end, negative_words, email, num_words, end_character


# noinspection DuplicatedCode
def vectorize_line_signature(line):
    line_lower = line.lower()
    pos_words = re.search(r'(?:regards|thanks|thank you|sincerely|good luck|cheers)', line_lower) is not None
    num_words = len(re.split(r'\s+', line))
    end_character = {':': 1, ';': 2, '"': 3, "'": 3, '?': 4, '!': 5, '.': 6}.get(line[-1], 0) if line else 0
    special_symbol_pattern = re.search(r'(?:[+*~_=#\-]{4,}|^--$)', line) is not None
    if re.match(r'^[^A-Z]+$', line):
        case = 1
    elif re.match(r'^[^a-z]+$', line):
        case = 2
    elif re.match(r'^[A-Z][^A-Z]*$', line):
        case = 3
    else:
        case = 4

    return pos_words, num_words, end_character, special_symbol_pattern, case


# noinspection DuplicatedCode
def vectorize_line_code(line):
    line_lower = line.lower()
    decl_keyword = re.search(r'(?:string|static|const|char|int|float|double|void|dim|typedef|struct|#include' +
                             r'|import|#define|#undef|#ifdef|#endif)', line_lower) is not None
    stmt_keyword_1 = re.search(r'(?:\w\+\+|\+\+\w|\+=)', line_lower) is not None
    stmt_keyword_2 = re.search(r'(?:if|else|elif|endif|done|do|switch|case)', line_lower) is not None
    stmt_keyword_3 = re.search(r'(?:while|do\s*{|for|foreach|done)', line_lower) is not None
    stmt_keyword_4 = re.search(r'(?:goto|continue;|next;|break;|last;|return)', line_lower) is not None

    eq_keyword_1 = re.search(r'(?:=|<=|<<=)', line_lower) is not None
    eq_keyword_2 = re.search(r'\w+\s*=\s*\w+[+/*-]\w+;', line_lower) is not None
    eq_keyword_3 = re.search(r'\w+\s*=\s*\w+\(\s*\w\s*,\s*\w\s*\);', line_lower) is not None
    eq_keyword_4 = re.search(r'\w+\s*=\s*\w;', line_lower) is not None

    func_def_1 = re.search(r'^\s*(?:sub|function|def|)', line_lower) is not None
    func_def_2 = re.search(r'^\s*(?:end sub|end function)', line_lower) is not None

    func_call = re.search(r'^\s*(?:^\s*[\w\d]+\((?:[^)][,\s])*\);)\s*$', line_lower) is not None

    brkt_1 = re.search(r'^\s*{', line_lower) is not None
    brkt_2 = re.search(r'{\s*$', line_lower) is not None
    brkt_3 = re.search(r'^\s*}', line_lower) is not None
    brkt_4 = re.search(r'}\s*$', line_lower) is not None

    cmnt_1 = re.search(r'^\s*//', line_lower) is not None
    cmnt_2 = re.search(r'^\s*/\*', line_lower) is not None
    cmnt_3 = re.search(r'\*/\s*$', line_lower) is not None

    end_char_1 = re.search(r';\s*$', line_lower) is not None
    end_char_2 = re.search(r'[?!]\s*$', line_lower) is not None

    return decl_keyword, stmt_keyword_1, stmt_keyword_2, stmt_keyword_3, stmt_keyword_4, \
           eq_keyword_1, eq_keyword_2, eq_keyword_3, eq_keyword_4, func_def_1, func_def_2, func_call, \
           brkt_1, brkt_2, brkt_3, brkt_4, cmnt_1, cmnt_2, cmnt_3, end_char_1, end_char_2


if __name__ == '__main__':
    main()
