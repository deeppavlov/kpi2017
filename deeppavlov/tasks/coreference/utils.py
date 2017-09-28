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
from collections import Counter
from sklearn.utils.linear_assignment_ import linear_assignment


def f1(p_num, p_den, r_num, r_den, beta=1):
    p = 0 if p_den == 0 else p_num / float(p_den)
    r = 0 if r_den == 0 else r_num / float(r_den)
    return 0 if p + r == 0 else (1 + beta * beta) * p * r / (beta * beta * p + r)

class CorefEvaluator(object):
    def __init__(self):
        self.evaluators = [Evaluator(m) for m in (muc, b_cubed, ceafe)]

    def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
        for e in self.evaluators:
            e.update(predicted, gold, mention_to_predicted, mention_to_gold)

    def get_f1(self):
        return sum(e.get_f1() for e in self.evaluators) / len(self.evaluators)

    def get_recall(self):
        return sum(e.get_recall() for e in self.evaluators) / len(self.evaluators)

    def get_precision(self):
        return sum(e.get_precision() for e in self.evaluators) / len(self.evaluators)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

class Evaluator(object):
    def __init__(self, metric, beta=1):
        self.p_num = 0
        self.p_den = 0
        self.r_num = 0
        self.r_den = 0
        self.metric = metric
        self.beta = beta

    def update(self, predicted, gold, mention_to_predicted, mention_to_gold):
        if self.metric == ceafe:
            pn, pd, rn, rd = self.metric(predicted, gold)
        else:
            pn, pd = self.metric(predicted, mention_to_gold)
            rn, rd = self.metric(gold, mention_to_predicted)
        self.p_num += pn
        self.p_den += pd
        self.r_num += rn
        self.r_den += rd

    def get_f1(self):
        return f1(self.p_num, self.p_den, self.r_num, self.r_den, beta=self.beta)

    def get_recall(self):
        return 0 if self.r_num == 0 else self.r_num / float(self.r_den)

    def get_precision(self):
        return 0 if self.p_num == 0 else self.p_num / float(self.p_den)

    def get_prf(self):
        return self.get_precision(), self.get_recall(), self.get_f1()

    def get_counts(self):
        return self.p_num, self.p_den, self.r_num, self.r_den


def evaluate_documents(documents, metric, beta=1):
    evaluator = Evaluator(metric, beta=beta)
    for document in documents:
        evaluator.update(document)
    return evaluator.get_precision(), evaluator.get_recall(), evaluator.get_f1()


def b_cubed(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        if len(c) == 1:
            continue

        gold_counts = Counter()
        correct = 0
        for m in c:
            if m in mention_to_gold:
                gold_counts[tuple(mention_to_gold[m])] += 1
        for c2, count in gold_counts.items():
            if len(c2) != 1:
                correct += count * count

        num += correct / float(len(c))
        dem += len(c)

    return num, dem


def muc(clusters, mention_to_gold):
    tp, p = 0, 0
    for c in clusters:
        p += len(c) - 1
        tp += len(c)
        linked = set()
        for m in c:
            if m in mention_to_gold:
                linked.add(mention_to_gold[m])
            else:
                tp -= 1
        tp -= len(linked)
    return tp, p


def phi4(c1, c2):
    return 2 * len([m for m in c1 if m in c2]) / float(len(c1) + len(c2))


def ceafe(clusters, gold_clusters):
    clusters = [c for c in clusters if len(c) != 1]
    scores = np.zeros((len(gold_clusters), len(clusters)))
    for i in range(len(gold_clusters)):
        for j in range(len(clusters)):
            scores[i, j] = phi4(gold_clusters[i], clusters[j])
    matching = linear_assignment(-scores)
    similarity = sum(scores[matching[:, 0], matching[:, 1]])
    return similarity, len(clusters), similarity, len(gold_clusters)


def lea(clusters, mention_to_gold):
    num, dem = 0, 0

    for c in clusters:
        if len(c) == 1:
            continue

        common_links = 0
        all_links = len(c) * (len(c) - 1) / 2.0
        for i, m in enumerate(c):
            if m in mention_to_gold:
                for m2 in c[i + 1:]:
                    if m2 in mention_to_gold and mention_to_gold[m] == mention_to_gold[m2]:
                        common_links += 1

        num += len(c) * common_links / float(all_links)
        dem += len(c)

    return num, dem

#
def conll2dict(iter_id, conll, agent, mode, epoch_done=False):
    data = {'doc_id': [],
            'part_id': [],
            'word_number': [],
            'word': [],
            'part_of_speech': [],
            'parse_bit': [],
            'lemma': [],
            'sense': [],
            'speaker': [],
            'entiti': [],
            'predict': [],
            'coreference': [],
            'iter_id': iter_id,
            'id': agent,
            'epoch_done': epoch_done,
            'mode': mode}

    with open(conll, 'r') as f:
        for line in f:
            row = line.split('\t')
            if row[0].startswith('#') or row[0] == '\n':
                pass
            else:
                assert len(row) >= 12
                data['doc_id'].append(row[0])
                data['part_id'].append(row[1])
                data['word_number'].append(row[2])
                data['word'].append(row[3])
                data['part_of_speech'].append(row[4])
                data['parse_bit'].append(row[5])
                data['lemma'].append(row[6])
                data['sense'].append(row[7])
                data['speaker'].append(row[8])
                data['entiti'].append(row[9])
                data['predict'].append(row[10])
                data['coreference'].append(row[11][0:-1])
        f.close()
    return data

def dict2conll(data, predict):
    #
    with open(predict, 'w') as CoNLL:
        for i in range(len(data['doc_id'])):
            if i == 0:
                CoNLL.write('#begin document ({}); part {}\n'.format(data['doc_id'][i], data["part_id"][i]))
                CoNLL.write(u'{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(data['doc_id'][i],
                                                    data["part_id"][i],
                                                    data["word_number"][i],
                                                    data["word"][i],
                                                    data["part_of_speech"][i],
                                                    data["parse_bit"][i],
                                                    data["lemma"][i],
                                                    data["sense"][i],
                                                    data["speaker"][i],
                                                    data["entiti"][i],
                                                    data["predict"][i],
                                                    data["coreference"][i]))
            elif i == len(data['doc_id'])-1:
                CoNLL.write(u'{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(data['doc_id'][i],
                                                    data["part_id"][i],
                                                    data["word_number"][i],
                                                    data["word"][i],
                                                    data["part_of_speech"][i],
                                                    data["parse_bit"][i],
                                                    data["lemma"][i],
                                                    data["sense"][i],
                                                    data["speaker"][i],
                                                    data["entiti"][i],
                                                    data["predict"][i],
                                                    data["coreference"][i]))
                CoNLL.write('\n')
                CoNLL.write('#end document\n')
            elif data['part_of_speech'][i] == 'SEN':
                CoNLL.write(u'{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(data['doc_id'][i],
                                                    data["part_id"][i],
                                                    data["word_number"][i],
                                                    data["word"][i],
                                                    data["part_of_speech"][i],
                                                    data["parse_bit"][i],
                                                    data["lemma"][i],
                                                    data["sense"][i],
                                                    data["speaker"][i],
                                                    data["entiti"][i],
                                                    data["predict"][i],
                                                    data["coreference"][i]))
                CoNLL.write('\n')
            else:
                if data['doc_id'][i] == data['doc_id'][i+1]:
                    CoNLL.write(u'{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(data['doc_id'][i],
                                                        data["part_id"][i],
                                                        data["word_number"][i],
                                                        data["word"][i],
                                                        data["part_of_speech"][i],
                                                        data["parse_bit"][i],
                                                        data["lemma"][i],
                                                        data["sense"][i],
                                                        data["speaker"][i],
                                                        data["entiti"][i],
                                                        data["predict"][i],
                                                        data["coreference"][i]))
                    if data["word_number"][i+1] == 0:
                        CoNLL.write('\n')
                else:
                    CoNLL.write(u'{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(data['doc_id'][i],
                                                        data["part_id"][i],
                                                        data["word_number"][i],
                                                        data["word"][i],
                                                        data["part_of_speech"][i],
                                                        data["parse_bit"][i],
                                                        data["lemma"][i],
                                                        data["sense"][i],
                                                        data["speaker"][i],
                                                        data["entiti"][i],
                                                        data["predict"][i],
                                                        data["coreference"][i]))
                    CoNLL.write('\n')
                    CoNLL.write('#end document\n')
                    CoNLL.write('#begin document ({}); part {}\n'.format(data['doc_id'][i+1], data["part_id"][i+1]))
        CoNLL.close()
    return None
