#!/usr/bin/env python3

from datetime import datetime
import json
import os
import re

import click
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks, layers, metrics, models

from tqdm import tqdm
import numpy as np

from util import util
from util.segmentation import load_fasttext_model, get_num_data_workers_and_queue_size, MailLinesSequence


logger = util.get_logger(__name__)

# Limit Tensorflow GPU memory
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
InteractiveSession(config=config)


TRAIN_BATCH_SIZE = 128                      # Training mini-batch size
INF_BATCH_SIZE = 256                        # Inference mini-batch size
LINE_LEN = 12                               # Maximum number of word tokens per line
INPUT_DIM = 100                             # Dimensionality of input embeddings
CONTEXT_SHAPE = (9, LINE_LEN, INPUT_DIM)    # Shape of the model context matrix (2*context+1, line_len, word_dim)


@click.group()
def main():
    """Train, apply, or evaluate an email or newsgroup message segmentation model."""
    pass


@main.command()
@click.argument('fasttext-model', type=click.Path(exists=True, dir_okay=False))
@click.argument('train-data', type=click.Path(exists=True, dir_okay=False))
@click.argument('output', type=click.Path(exists=False))
@click.option('-l', '--loss-function', type=click.Choice(['categorical_crossentropy', 'categorical_hinge']),
              default='categorical_crossentropy')
@click.option('-v', '--validation-data', help='Validation Data JSON')
@click.option('-f', '--fine-tune', help='Only fine-tune the given pre-trained model',
              type=click.Path(exists=True, dir_okay=False))
@click.option('-t', '--tensorboard', is_flag=True, help='Tensorboard log data directory')
def train(fasttext_model, train_data, output, **kwargs):
    """
    Train message segmenter to classify lines of an email or newsgroup message.

    Arguments:
        fasttext_model: pre-trained FastText embedding
        train_data: input training data as JSON
        output: Model output
    """
    logger.info('Loading FastText model...')
    load_fasttext_model(fasttext_model)
    train_model(train_data, output, **kwargs)


@main.command()
@click.argument('model', type=click.Path(exists=True, dir_okay=False))
@click.argument('fasttext-model', type=click.Path(exists=True, dir_okay=False))
@click.argument('test-data', type=click.File('r'))
@click.option('-o', '--output-json', help='Output JSONL file', type=click.File('w'))
def predict(model, fasttext_model, test_data, **kwargs):
    """
    Apply trained message segmentation model to predict lines of an email or newsgroup message.

    Arguments:
        model: Trained HDF5 segmenter model
        fasttext_model: pre-trained FastText embedding
        test_data: test message dump as JSON
    """
    output_json = kwargs.get('output_json')

    logger.info('Loading FastText model...')
    load_fasttext_model(fasttext_model)

    segmenter = models.load_model(model)
    to_stdout = output_json is None

    num_workers, queue_size = get_num_data_workers_and_queue_size()

    logger.info('Predicting {}...'.format(test_data.name))
    while True:
        # Do not load more than 1k lines at once
        pred_seq = MailLinesSequence(test_data, CONTEXT_SHAPE, labeled=False, batch_size=INF_BATCH_SIZE, max_lines=1000)
        if len(pred_seq) == 0:
            break

        predictions = segmenter.predict_generator(pred_seq,
                                                  verbose=(not to_stdout),
                                                  steps=(None if not to_stdout else 10),
                                                  use_multiprocessing=True,
                                                  workers=num_workers,
                                                  max_queue_size=queue_size)
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
    Evaluate a trained message segmentation model.

    Arguments:
        model: Trained HDF5 segmenter model
        fasttext_model: pre-trained FastText embedding
        eval_data: test message dump as JSON
    """
    logger.info('Loading FastText model...')
    load_fasttext_model(fasttext_model)

    def _binarize_pred_tensors(cls, *tensors):
        """Binarize multi-class prediction tensors to the given class."""
        cls = MailLinesSequence.LABEL_MAP[cls] if type(cls) is str else cls
        return [K.cast(K.equal(K.argmax(t, axis=-1), cls), t.dtype) for t in tensors]

    def quotation_accuracy(y_true, y_pred):
        return metrics.binary_accuracy(*_binarize_pred_tensors('quotation', y_true, y_pred))

    def patch_accuracy(y_true, y_pred):
        return metrics.binary_accuracy(*_binarize_pred_tensors('patch', y_true, y_pred))

    def paragraph_accuracy(y_true, y_pred):
        return metrics.binary_accuracy(*_binarize_pred_tensors('paragraph', y_true, y_pred))

    def log_data_accuracy(y_true, y_pred):
        return metrics.binary_accuracy(*_binarize_pred_tensors('log_data', y_true, y_pred))

    def mua_signature_accuracy(y_true, y_pred):
        return metrics.binary_accuracy(*_binarize_pred_tensors('mua_signature', y_true, y_pred))

    def personal_signature_accuracy(y_true, y_pred):
        return metrics.binary_accuracy(*_binarize_pred_tensors('personal_signature', y_true, y_pred))

    segmenter = models.load_model(model)
    segmenter.compile(optimizer=segmenter.optimizer, loss=segmenter.loss,
                      metrics=['categorical_accuracy',
                               quotation_accuracy, patch_accuracy, paragraph_accuracy,
                               log_data_accuracy, mua_signature_accuracy, personal_signature_accuracy])

    logger.info('Evaluating {}...'.format(eval_data.name))
    eval_seq = MailLinesSequence(eval_data, CONTEXT_SHAPE, labeled=True, batch_size=INF_BATCH_SIZE)

    cls_sum = np.zeros(len(MailLinesSequence.LABEL_MAP))
    for _, labels in tqdm(eval_seq, desc='Counting labels', unit='samples', leave=False):
        cls_sum = np.add(cls_sum, np.sum(labels, axis=0))

    click.echo('Ground-truth class distribution:')
    cls_sum /= len(eval_seq) * INF_BATCH_SIZE
    cls_sum = sorted(enumerate(cls_sum), key=lambda x: x[1], reverse=True)
    for i, prob in cls_sum:
        click.echo(' {: >19}: {:.4f}'.format(MailLinesSequence.LABEL_MAP_INVERSE[i], prob))

    logger.info('Predicting samples...')
    num_workers, queue_size = get_num_data_workers_and_queue_size()
    segmenter.evaluate_generator(eval_seq, verbose=True,
                                 use_multiprocessing=True, workers=num_workers, max_queue_size=queue_size)


def train_model(training_data, output_model, loss_function='categorical_crossentropy',
                validation_data=None, fine_tune=None, tensorboard=False):
    """
    Train message segmentation model.

    :param training_data: JSON file with training data
    :param output_model: path prefix for model checkpoints
    :param loss_function: optimization loss function
    :param validation_data: JSON file with validation data
    :param fine_tune: fine-tune model from given file instead of training from scratch
    :param tensorboard: Tensorboard log data directory
    """

    tb_callback = callbacks.TensorBoard(log_dir='./data/graph/' + str(datetime.now()), update_freq='batch',
                                        write_graph=False, write_images=False)
    es_callback = callbacks.EarlyStopping(monitor='val_loss', verbose=1, patience=5)
    cp_callback = callbacks.ModelCheckpoint(output_model + '.epoch-{epoch:02d}.val_loss-{val_loss:.3f}.h5')
    cp_callback_no_val = callbacks.ModelCheckpoint(output_model + '.epoch-{epoch:02d}.loss-{loss:.3f}.h5')

    def get_line_model():
        line_input = layers.Input(shape=(LINE_LEN, INPUT_DIM))
        masking = layers.Masking(0)(line_input)
        bi_seq = layers.Bidirectional(layers.GRU(128), merge_mode='sum')(masking)
        bi_seq = layers.BatchNormalization()(bi_seq)
        bi_seq = layers.Activation('relu')(bi_seq)
        return line_input, bi_seq

    def get_context_model():
        context_input = layers.Input(shape=CONTEXT_SHAPE)
        conv2d = layers.Conv2D(128, (4, 4))(context_input)
        conv2d = layers.BatchNormalization()(conv2d)
        conv2d = layers.Activation('relu')(conv2d)
        conv2d = layers.Conv2D(128, (3, 3))(conv2d)
        conv2d = layers.Activation('relu')(conv2d)
        conv2d = layers.MaxPooling2D(2)(conv2d)
        flatten = layers.Flatten()(conv2d)
        dense = layers.Dense(128)(flatten)
        dense = layers.Activation('relu')(dense)
        return context_input, dense

    def get_base_model():
        line_input_cur, line_model_cur = get_line_model()
        line_input_prev, line_model_prev = get_line_model()
        context_input, context_model = get_context_model()

        concat = layers.concatenate([line_model_cur, line_model_prev, context_model])
        dropout = layers.Dropout(0.25)(concat)
        dense = layers.Dense(len(MailLinesSequence.LABEL_MAP))(dropout)
        output = layers.Activation('softmax')(dense)

        return models.Model(inputs=[line_input_cur, line_input_prev, context_input], outputs=output)

    if fine_tune is None:
        segmenter = get_base_model()
    else:
        segmenter = models.load_model(fine_tune)
        logger.info('Freezing layers...')

        for layer in segmenter.layers[:-2]:
            layer.trainable = False

    compile_args = {
        'optimizer': 'adam',
        'loss': loss_function,
        'metrics': ['categorical_accuracy']
    }

    effective_callbacks = []
    if fine_tune is None:
        effective_callbacks = [es_callback, cp_callback] if validation_data is not None else [cp_callback_no_val]
    if tensorboard:
        effective_callbacks.append(tb_callback)

    segmenter.compile(**compile_args)
    segmenter.summary()

    num_workers, queue_size = get_num_data_workers_and_queue_size()
    train_seq = MailLinesSequence(training_data, CONTEXT_SHAPE, labeled=True, batch_size=TRAIN_BATCH_SIZE)
    val_seq = MailLinesSequence(validation_data, CONTEXT_SHAPE, labeled=True,
                                batch_size=INF_BATCH_SIZE) if validation_data else None

    segmenter.fit_generator(train_seq, epochs=20, validation_data=val_seq, shuffle=True, use_multiprocessing=True,
                            workers=num_workers, max_queue_size=queue_size, callbacks=effective_callbacks)

    if fine_tune is not None:
        logger.info('Unfreezing layers...')
        for layer in segmenter.layers:
            layer.trainable = True

        effective_callbacks.append(cp_callback_no_val)
        segmenter.compile(**compile_args)
        segmenter.fit_generator(train_seq, epochs=5, validation_data=val_seq, shuffle=True, use_multiprocessing=True,
                                workers=num_workers, max_queue_size=queue_size, callbacks=effective_callbacks)


def predict_raw_text(segmentation_model, message):
    """
    Predict segments of raw message text.

    :param segmentation_model: Trained segmentation model
    :param message: email message text
    :return: Generator of (message text, label text)
    """

    pred_seq = MailLinesSequence(message, CONTEXT_SHAPE, labeled=False, input_is_raw_text=True,
                                 batch_size=INF_BATCH_SIZE)
    return _post_process_labels(pred_seq, segmentation_model.predict_generator(pred_seq))


def reformat_raw_text_recursive(segmentation_model, email, exclude_classes=None, max_depth=10):
    """
    Predicts and recursively reformats an email.
    Nested quotations will be parsed, quotation markers and symbols are removed and
    the contents are predicted again.

    :param segmentation_model: trained segmentation model
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
        predictions = predict_raw_text(segmentation_model, text)
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


def _post_process_labels(mails_sequence, labels_softmax):
    """
    Postprocess predicted lines to replace softmax vectors with txt labels
    and clean up some prediction noise.

    :param mails_sequence: input MailLinesSequence
    :param labels_softmax: predicted labels as softmax vectors for lines in `mails_sequence`
    :return: Generator of (line text, label text)
    """

    lines = mails_sequence.mail_lines
    context_size = CONTEXT_SHAPE[0] // 2

    for i, (line, label) in enumerate(zip(lines, labels_softmax)):
        line = line if not mails_sequence.labeled else line[0]
        label_argmax = np.argmax(label)
        label_argsort = np.argsort(label)[::-1]
        label_text = MailLinesSequence.LABEL_MAP_INVERSE[label_argmax]

        prev_l = []
        for j in range(context_size):
            if i - j < 0 or i - j + 1 in mails_sequence.mail_start_indices:
                break
            prev_l.append(MailLinesSequence.LABEL_MAP_INVERSE[np.argmax(labels_softmax[i - j])])
        prev_l.extend([None] * (context_size - len(prev_l)))
        prev_l.reverse()

        next_l = []
        for j in range(context_size):
            if i + 1 + j >= len(lines) or i + 1 + j in mails_sequence.mail_end_indices:
                break
            next_l.append(MailLinesSequence.LABEL_MAP_INVERSE[np.argmax(labels_softmax[i + 1 + j])])
        next_l.extend([None] * (context_size - len(next_l)))

        empty_classes = ['<empty>', 'visual_separator', None]
        prev_set_no_blank = set([l for l in prev_l if l not in empty_classes])
        next_set_no_blank = set([l for l in next_l if l not in empty_classes])

        if line is None or label is None:
            yield '<PAD>\n', None
            labels_softmax[i] = -1
            continue

        # Correct <empty>
        if line.strip() == '':
            label_text = '<empty>'

        # Empty lines have to be empty
        elif label_text == '<empty>' and line.strip() != '':
            label_text = prev_l[-1] if prev_l[-1] not in empty_classes else 'paragraph'

        elif label_text == 'visual_separator':
            # Never change separators
            pass

        # Bleeding quotations
        elif label_text == 'quotation' and prev_l[-1] == 'quotation' \
                and lines[i - 1].strip() and lines[i - 1].strip() \
                and next_l[0] != 'quotation' and lines[i - 1].strip()[0] != line.strip()[0] \
                and prev_l[-1] not in empty_classes:
            label_text = prev_l[-1]

        # Quotations
        elif label_text != 'quotation' \
                and (line.strip().startswith('>') or line.strip().startswith('|')) \
                and (MailLinesSequence.LABEL_MAP['quotation'] in label_argsort[:3] or prev_l[-1] == 'quotation'):
            label_text = 'quotation'

        # Quotation markers
        elif label_text == 'quotation' and prev_l[-1] in empty_classes \
                and mails_sequence.LABEL_MAP['quotation_marker'] in label_argsort[:3]:
            label_text = 'quotation_marker'

        # Interrupted short blocks
        elif label_text != prev_l[-1] and next_l[0] == prev_l[-1] \
                and prev_l[-1] in ['closing', 'personal_signature', 'mua_signature', 'inline-header', 'technical']:
            label_text = prev_l[-1]

        # Interrupted long blocks
        elif len(prev_set_no_blank) == 1 and label_text != [*prev_set_no_blank][0] and \
                [*prev_set_no_blank][0] in next_set_no_blank \
                and [*prev_set_no_blank][0] in ['mua_signature', 'personal_signature',
                                                'patch', 'code', 'tabular', 'technical'] \
                and mails_sequence.LABEL_MAP[[*prev_set_no_blank][0]] == label_argsort[1]:
            label_text = [*prev_set_no_blank][0]

        # Interrupting stray classes
        elif label_text in ['technical', 'mua_signature', 'personal_signature', 'patch', 'tabular'] \
                and prev_l[-1] != label_text and prev_l[-1] not in empty_classes \
                and (next_l[0] == prev_l[-1] or (next_l[1] == prev_l[-1] and next_l[0] in empty_classes)):
            label_text = prev_l[-1]

        labels_softmax[i] = mails_sequence.LABEL_MAP_ONEHOT[label_text]
        yield line, label_text


def export_mail_annotation_spans(predictions_softmax, pred_sequence, output_file=None, verbose=True):
    """
    Export predicted lines to JSON (start, end) spans.

    :param predictions_softmax: predicted labels as softmax vectors
    :param pred_sequence: input line sequence
    :param output_file: output JSON file
    :param verbose: print labeled lines to STDOUT
    """

    text = ''
    main_content = ''
    annotations = []
    prev_label = None
    cur_label = None
    start_offset = 0
    mail_dict = None

    def write_annotations(d, a, m=None):
        """Write annotations to JSON file."""
        if not a or 'text' not in d or not d['text']:
            return

        d = {k: d[k] for k in d if k != 'id'}
        d.update({'labels': a, 'text': d['text'].lstrip()})
        if 'meta' not in d:
            d['meta'] = {}
        d['meta']['main_content'] = m

        json.dump(d, output_file)
        output_file.write('\n')

    for i, (line, label_text) in enumerate(_post_process_labels(pred_sequence, predictions_softmax)):
        cur_label = label_text
        if prev_label is None:
            prev_label = cur_label

        if i in pred_sequence.mail_start_indices:
            if verbose:
                click.echo(' {0:>>20}    --->    <<< MAIL START >>>'.format(''))
            mail_dict = pred_sequence.mail_metadata_map.get(i, {})

        cur_offset = len(text) - 1
        text += line
        text = text.lstrip()
        if cur_label in ['paragraph', 'section_heading']:
            main_content += line

        if i in pred_sequence.mail_end_indices:
            if output_file:
                if prev_label not in [None, '<empty>']:
                    annotations.append((start_offset, cur_offset, prev_label))
                write_annotations(mail_dict, annotations, main_content)

            mail_dict = None
            annotations.clear()
            start_offset = 0
            prev_label = None
            text = ''
            main_content = ''
            continue

        if verbose:
            click.echo(' {:>20}    --->    {}'.format(label_text, line), nl=False)

        if cur_label != prev_label:
            if output_file and prev_label not in [None, '<empty>']:
                annotations.append((start_offset, cur_offset, prev_label))

            start_offset = cur_offset + 1
            prev_label = cur_label

    if output_file and mail_dict:
        if cur_label not in ['<empty>', None]:
            annotations.append((start_offset, len(text) - 1, cur_label))
        write_annotations(mail_dict, annotations, main_content)


if __name__ == '__main__':
    main()
