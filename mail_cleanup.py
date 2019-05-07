import json
import numpy as np
import os
import pickle
import plac
import re
from sklearn import svm
import sys

CONTENT = 0
HEADER = 1
SIGNATURE = 2
CODE = 3

label_map = {
    'closing': CONTENT,
    'inline_headers': HEADER,
    'log_data': CONTENT,
    'mua_signature': SIGNATURE,
    'paragraph': CONTENT,
    'patch': CODE,
    'personal_signature': SIGNATURE,
    'quotation': CONTENT,
    'quotation_marker': HEADER,
    'raw_code': CODE,
    'salutation': CONTENT,
    'section_heading': CONTENT,
    'stacktrace': CODE,
    'tabular': CONTENT,
    'technical': CODE,
    'visual_separator': SIGNATURE,
}


@plac.annotations(
    cmd=('Command', 'positional', None, str, None, 'CMD'),
    input_file=('Input JSONL file', 'positional', None, str, None, 'FILE'),
    model_dir=('Model directory', 'positional', None, str, None, 'DIR')
)
def main(cmd, input_file, model_dir):
    labeled_mails = []
    unlabeled_mails = []
    for line in open(input_file).readlines():
        mail_json = json.loads(line)
        if not mail_json['annotations']:
            unlabeled_mails.append([l for l in mail_json['text'].split('\n') if not check_if_quotation_line(l)])
            continue

        labeled_mails.append([l for l in label_lines(mail_json) if not check_if_quotation_line(l[0])])
    print('Labeled:', len(labeled_mails))
    print('Unlabeled:', len(unlabeled_mails))

    if cmd == 'train':
        train_clf(labeled_mails, model_dir)
    elif cmd == 'predict':
        predict(unlabeled_mails, model_dir)
    else:
        print('Invalid command.', file=sys.stderr)
        exit(1)


def check_if_quotation_line(line):
    return re.match(r'^\s*[>|]', line) is not None


def train_clf(labeled_mails, model_dir):
    X_header, y_header = concat_vectors(labeled_mails, vectorize_line_header)
    X_signature, y_signature = concat_vectors(labeled_mails, vectorize_line_signature)
    X_code, y_code = concat_vectors(labeled_mails, vectorize_line_code)

    y_header_start, y_header_end = get_boundary_labels(y_header, HEADER)
    y_signature_start, y_signature_end = get_boundary_labels(y_signature, SIGNATURE)
    y_code_start, y_code_end = get_boundary_labels(y_code, CODE)

    print('Training header start model...')
    clf_header_start = svm.SVC(kernel='poly', gamma='scale')
    clf_header_start.fit(X_header, y_header_start)
    pickle.dump(clf_header_start, open(os.path.join(model_dir, 'header_start.model'), 'wb'))

    print('Training header end model...')
    clf_header_end = svm.SVC(kernel='poly', gamma='scale')
    clf_header_end.fit(X_header, y_header_end)
    pickle.dump(clf_header_end, open(os.path.join(model_dir, 'header_end.model'), 'wb'))

    print('Training signature start model...')
    clf_signature_start = svm.SVC(kernel='poly', gamma='scale')
    clf_signature_start.fit(X_signature, y_signature_start)
    pickle.dump(clf_signature_start, open(os.path.join(model_dir, 'signature_start.model'), 'wb'))

    print('Training signature end model...')
    clf_signature_end = svm.SVC(kernel='poly', gamma='scale')
    clf_signature_end.fit(X_signature, y_signature_end)
    pickle.dump(clf_signature_end, open(os.path.join(model_dir, 'signature_end.model'), 'wb'))

    print('Training code start model...')
    clf_code_start = svm.SVC(kernel='poly', gamma='scale')
    clf_code_start.fit(X_code, y_code_start)
    pickle.dump(clf_code_start, open(os.path.join(model_dir, 'code_start.model'), 'wb'))

    print('Training code end model...')
    clf_code_end = svm.SVC(kernel='poly', gamma='scale')
    clf_code_end.fit(X_code, y_code_end)
    pickle.dump(clf_code_end, open(os.path.join(model_dir, 'code_end.model'), 'wb'))


def predict(unlabeled_mails, model_dir):
    clf_header_start = pickle.load(open(os.path.join(model_dir, 'header_start.model'), 'rb'))
    clf_header_end = pickle.load(open(os.path.join(model_dir, 'header_end.model'), 'rb'))
    clf_signature_start = pickle.load(open(os.path.join(model_dir, 'signature_start.model'), 'rb'))
    clf_signature_end = pickle.load(open(os.path.join(model_dir, 'signature_end.model'), 'rb'))
    clf_code_start = pickle.load(open(os.path.join(model_dir, 'code_start.model'), 'rb'))
    clf_code_end = pickle.load(open(os.path.join(model_dir, 'code_end.model'), 'rb'))

    for mail in unlabeled_mails:
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
        for i, l in enumerate(mail):
            if y_header_start[i] == 1:
                in_header = True
            if y_signature_start[i] == 1:
                in_signature = True
            if y_code_start[i] == 1:
                in_code = True

            if i > 0 and y_header_end[i - 1] == 1:
                in_header = False
            if i > 0 and y_signature_end[i - 1] == 1:
                in_signature = False
            if i > 0 and y_code_end[i - 1] == 1:
                in_code = False

            cls = CONTENT
            if in_header:
                cls = HEADER
            if in_signature:
                cls = SIGNATURE
            if in_code:
                cls = CODE

            mail[i] = (l, cls)
            print(mail[i])


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
    in_header = False
    for i, l in enumerate(y):
        if y[i] == filter_label and not in_header:
            labels_start[i] = 1
            in_header = True
        elif i > 0 and in_header and y[i] != filter_label and y[i - 1] == filter_label:
            labels_end[i - 1] = 1

        if y[i] != filter_label:
            in_header = False
    return labels_start, labels_end


def label_lines(doc):
    lines = doc['text'].split('\n')
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
            yield l, None
            continue

        if offset <= annotations[-1]['end_offset'] and end_offset >= annotations[-1]['start_offset']:
            yield l, label_map[annotations[-1]['label']]
        else:
            yield l, None

        offset = end_offset


def vectorize_lines(lines, callback):
    vectors = []
    empty_lines = 0
    labels = np.zeros(len(lines))

    for i, l_tup in enumerate(lines):
        if type(l_tup) is tuple:
            l, labels[i] = l_tup
            if l_tup[1] is None:
                labels[i] = CONTENT
        else:
            l = l_tup

        if not l.strip():
            empty_lines += 1
        else:
            empty_lines = 0

        first_line = int(i == 0)
        last_line = int(i == len(lines) - 1)

        vectors.append((empty_lines, first_line, last_line) + callback(l))

    matrix = np.zeros((len(lines), len(vectors[0]) * 3))
    for i, v in enumerate(vectors):
        prev_line = vectors[i - 1] if i != 0 else (0,) * len(v)
        next_line = vectors[i + 1] if i != len(vectors) - 1 else (0,) * len(v)
        matrix[i] = prev_line + v + next_line

    return matrix, labels


def vectorize_line_header(line):
    line_lower = line.lower()
    pos_word_start = re.search(r'^(?:from:|re:|aw:|fwd:|in article|in message)', line_lower) is not None
    pos_word_contains = re.search(r'(?:original message|fwd:|in reply to)', line_lower) is not None
    pos_word_end = re.search(r'(?:wrote:|said:|writes|commented:)$', line_lower) is not None
    negative_words = re.search(r'(?:hi|hello|hey|dear|regards|thanks|thank you|sincerely|good luck|cheers)', line_lower) is not None
    email = re.search(r'^([a-zA-Z0-9_\-\.]+)@((\[[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.)|(([a-zA-Z0-9\-]+\.)+))([a-zA-Z]{2,4}|[0-9]{1,3})(\]?)$', line_lower) is not None
    num_words = len(re.split(r'\s+', line))
    end_character = {':': 1, ';': 2, '"': 3, "'": 3, '?': 4, '!': 5, '.': 6}.get(line[-1], 0) if line else 0

    return pos_word_start, pos_word_contains, pos_word_end, negative_words, email, num_words, end_character


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


def vectorize_line_code(line):
    line_lower = line.lower()
    decl_keyword = re.search(r'(?:string|static|const|char|int|float|double|void|dim|typedef|struct|#include|import|#define|#undef|#ifdef|#endif)', line_lower) is not None
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
    plac.call(main)
