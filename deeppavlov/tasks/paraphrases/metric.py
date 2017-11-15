"""
Copyright 2017 Neural Networks and Deep Learning lab, MIPT

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np


def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.
    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + 10e-8)
    return precision


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.
    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + 10e-8)
    return recall


def fbeta_score(y_true, y_pred, beta=1):
    """Computes the F score.

    The F score is the weighted harmonic mean of precision and recall.
    Here it is only computed as a batch-wise average, not globally.
    This is useful for multi-label classification, where input samples can be
    classified as sets of labels. By only using accuracy (precision) a model
    would achieve a perfect score by simply assigning every class to every
    input. In order to avoid this, a metric should penalize incorrect class
    assignments as well (recall). The F-beta score (ranged from 0.0 to 1.0)
    computes this, as a weighted mean of the proportion of correct class
    assignments vs. the proportion of incorrect class assignments.
    With beta = 1, this is equivalent to a F-measure. With beta < 1, assigning
    correct classes becomes more important, and with beta > 1 the metric is
    instead weighted towards penalizing incorrect class assignments.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')

    # If there are no true positives, fix the F score at 0 like sklearn.
    if np.sum(np.round(np.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + 10e-8)
    return fbeta_score


def accuracy(y_true, y_pred):
    """Accuracy metric."""

    if len(y_pred) > 0:
        return np.equal(y_true, y_pred).astype(int).mean()
    return 0


class BinaryClassificationMetrics(object):
    """The class converts text representations of predictions and labels to binary ones and calculate metrics on them.

    Attributes:
        true_str: text representation of the positive answer
        y_pred: predictions of a model
        y_true: labels
    """

    def __init__(self, true_str):
        """Initialize predictions and labels with empty lists."""

        self.true_str = true_str
        self.y_pred = []
        self.y_true = []

    def clear(self):
        """Reset predictions and labels."""

        del self.y_pred[:]
        del self.y_true[:]

    def update(self, observation, y):
        """Update predictions and labels according with an observation."""

        if y and 'text' in observation:
            self.y_pred.append(1 if observation['text'] == self.true_str else 0)
            self.y_true.append(1 if y[0] == self.true_str else 0)

    def report(self):
        """Calculate metrics and return the result."""

        if len(self.y_pred) > 0:
            report = {
                'f1': fbeta_score(self.y_true, self.y_pred),
                'accuracy': accuracy(self.y_true, self.y_pred),
                'cnt': len(self.y_pred)
            }
            return report
        return dict()
