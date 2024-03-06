from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score, \
    f1_score, recall_score, precision_score
from tabulate import tabulate
from sklearn.preprocessing import LabelEncoder
import numpy as np
import logging
import io
import requests

ACCURACY_SCORE = 'accuracy'
F1_UNWEIGHTED = 'f1_unweighted'
F1_WEIGHTED = 'f1_weighted'
RECALL_UNWEIGHTED = 'recall_unweighted'
RECALL_WEIGHTED = 'recall_weighted'
PRECISION_UNWEIGHTED = 'precision_unweighted'
PRECISION_WEIGHTED = 'precision_weighted'
TOP2_SCORE = 'top_2'
TOP5_SCORE = 'top_5'


def pretty_print_confusion_matrix(target, predicted, label_map):
    """
    Returns a nicely formatted confusion matrix ready for printing
    :return:
    """

    cnf_mat = confusion_matrix(target, predicted)

    table = [[] for idx in range(len(label_map))]

    labels = [None for idx in range(len(label_map))]

    for key, label in label_map.items():
        idx = int(key)
        table[idx] = [label] + cnf_mat[idx, :].tolist()
        labels[idx] = label

    return cnf_mat, tabulate(table, headers=labels)


def top_n_score(y_target, y_pred, class_labels, top_n=10, weighted=True):
    """
    This method returns the top_n_score.
    The top_n_score returns 1 if the target_label is in the first n predictions in y_pred
    :param y_target: This is an array of target labels with shape (n_samples,)
    :param y_pred: This is an array of predicted probabilities with shape (n_samples, n_classes)
    :param class_labels: This is the list of all possible classes
    :param top_n: This determines how many predictions to consider
    :param weighted: Return the raw score or weighted by the number of samples
    :return:
    """

    if top_n >= len(class_labels):
        top_n -= 1

    labelbin = LabelEncoder()
    labelbin.fit(class_labels)

    encoded_labels = labelbin.transform(y_target)

    sorted_prob = np.argsort(-y_pred, axis=1)

    top_n_predictions = sorted_prob[:, :top_n]
    encoded_probability = np.take_along_axis(y_pred, encoded_labels[:, None], axis=1)
    encoded_probability = encoded_probability.reshape(encoded_probability.shape[0], )

    bool_top_n = encoded_labels.reshape(-1, 1) == top_n_predictions
    bool_top_n = np.sum(bool_top_n, axis=1).astype(np.bool_)
    combined = np.logical_and(bool_top_n, (encoded_probability > 0))

    score = sum(combined) if not weighted else sum(combined)/combined.shape[0]
    return score


def get_tracked_metrics(classes, metric_name=None):

    accuracy_scorer = make_scorer(accuracy_score)
    f1_scorer_unweighted = make_scorer(f1_score, average='macro')
    f1_scorer_weighted = make_scorer(f1_score, average='weighted')
    recall_scorer_unweighted = make_scorer(recall_score, average='macro')
    recall_scorer_weighted = make_scorer(recall_score, average='weighted')
    precision_scorer_unweighted = make_scorer(precision_score, average='macro', zero_division=1)
    precision_scorer_weighted = make_scorer(precision_score, average='weighted', zero_division=1)
    top_2_scorer = make_scorer(top_n_score, needs_proba=True, class_labels=classes, top_n=2)
    top_5_scorer = make_scorer(top_n_score, needs_proba=True, class_labels=classes, top_n=5)

    metrics = {
        'top_2': top_2_scorer,
        'top_5': top_5_scorer,
        'accuracy': accuracy_scorer,
        'f1_unweighted': f1_scorer_unweighted,
        'f1_weighted': f1_scorer_weighted,
        'recall_unweighted': recall_scorer_unweighted,
        'recall_weighted': recall_scorer_weighted,
        'precision_unweighted': precision_scorer_unweighted,
        'precision_weighted': precision_scorer_weighted,
    }


    if metric_name is not None:
        if type(metric_name) is list:
            return {key: metrics.get(key, None) for key in metric_name}

        return metrics.get(metric_name, None)

    return metrics


class Logger(object):

    def __init__(self, logger_name, active=True, telegram_chat_id=None):

        self.logger_name = logger_name
        self.stream = io.StringIO()
        self.handler = logging.StreamHandler(self.stream)
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)
        self.active = active
        self.telegram_chat_id = telegram_chat_id

        for handler in self.logger.handlers:
            self.logger.removeHandler(handler)

        self.logger.addHandler(self.handler)

    def log(self, message, level=logging.DEBUG, to_telegram=True):
        if not self.active:
            return True
        message = "%s: %s" % (self.logger_name, message)
        self.logger.log(level, message)

        if to_telegram and self.telegram_chat_id is not None:
            send_message_telegram(message, self.telegram_chat_id)
        return True

    def to_string(self):
        self.handler.flush()

        return self.stream.getvalue()


def send_message_telegram(message, telegram_chat_id=None):
    if telegram_chat_id is None:
        # do nothing
        return
    payload = {
        'chat_id': telegram_chat_id,
        'text': message
    }

    url = BASE_URL + "sendMessage"
    try:
        requests.get(url, params=payload)
    except Exception:
        pass

    return True
