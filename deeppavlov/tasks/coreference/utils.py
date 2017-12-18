# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import time
import os
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm
import parlai.core.build_data as build_data
import tensorflow as tf
from collections import defaultdict
import numpy as np
import uuid


class watcher():
    """
    A class that ensures that there are no violations in the structure of mentions.
    """
    def __init__(self):
        self.mentions = 0
    
    def mentions_closed(self, s):
        s = s.split('|')
        for x in s:
            if x[0] == '(' and x[-1] != ')':
                self.mentions += 1
            elif x[0] != '(' and x[-1] == ')':
                self.mentions -= 1
                
        if self.mentions != 0:
            return False
        else:
            return True


def RuCoref2CoNLL(path, out_path, language='russian'):
    """
    RuCor corpus files are converted to standard conll format.
    Args:
        path: path to the RuCorp dataset
        out_path: -
        language: language of dataset

    Returns: Nothing

    """
    data = {"doc_id": [],
            "part_id": [],
            "word_number": [],
            "word": [],
            "part_of_speech": [],
            "parse_bit": [],
            "lemma": [],
            "sense": [],
            "speaker": [],
            "entiti": [],
            "predict": [],
            "coref": []}

    part_id = '0'
    speaker = 'spk1'
    sense = '-'
    entiti = '-'
    predict = '-'

    tokens_ext = "txt"
    groups_ext = "txt"
    tokens_fname = "Tokens"
    groups_fname = "Groups"

    tokens_path = os.path.join(path, ".".join([tokens_fname, tokens_ext]))
    groups_path = os.path.join(path, ".".join([groups_fname, groups_ext]))
    print('Convert rucoref corpus into conll format ...')
    start = time.time()
    coref_dict = {}
    with open(groups_path, "r") as groups_file:
        for line in groups_file:
            doc_id, variant, group_id, chain_id, link, shift, lens, content, tk_shifts, attributes, head, hd_shifts = line[:-1].split('\t')

            if doc_id not in coref_dict:
                coref_dict[doc_id] = {'unos': defaultdict(list), 'starts': defaultdict(list), 'ends': defaultdict(list)}

            if len(tk_shifts.split(',')) == 1:
                coref_dict[doc_id]['unos'][shift].append(chain_id)
            else:
                tk = tk_shifts.split(',')
                coref_dict[doc_id]['starts'][tk[0]].append(chain_id)
                coref_dict[doc_id]['ends'][tk[-1]].append(chain_id)
        groups_file.close()

    # Write conll structure
    with open(tokens_path, "r") as tokens_file:
        k = 0
        doc_name = '0'
        for line in tokens_file:
            doc_id, shift, length, token, lemma, gram = line[:-1].split('\t')
            
            if doc_id == 'doc_id':
                continue
            
            if doc_id != doc_name:
                doc_name = doc_id
                w = watcher()
                k = 0
                
            data['word'].append(token)    
            data['doc_id'].append(doc_id)
            data['part_id'].append(part_id)
            data['lemma'].append(lemma)
            data['sense'].append(sense)
            data['speaker'].append(speaker)
            data['entiti'].append(entiti)
            data['predict'].append(predict)
            data['parse_bit'].append('-')

            opens = coref_dict[doc_id]['starts'][shift] if shift in coref_dict[doc_id]['starts'] else []
            ends = coref_dict[doc_id]['ends'][shift] if shift in coref_dict[doc_id]['ends'] else []
            unos = coref_dict[doc_id]['unos'][shift] if shift in coref_dict[doc_id]['unos'] else []
            s = []
            s += ['({})'.format(el) for el in unos]
            s += ['({}'.format(el) for el in opens]
            s += ['{})'.format(el) for el in ends]
            s = '|'.join(s)
            if len(s) == 0:
                s = '-'
                data['coref'].append(s)
            else:
                data['coref'].append(s)
            
            closed = w.mentions_closed(s)
            if gram == 'SENT' and not closed:
                data['part_of_speech'].append('.')
                data['word_number'].append(k)
                k += 1 
                
            elif gram == 'SENT' and closed:
                data['part_of_speech'].append(gram)
                data['word_number'].append(k)
                k = 0
            else:
                data['part_of_speech'].append(gram)
                data['word_number'].append(k)
                k += 1
       
        tokens_file.close()

    # Write conll structure in file
    conll = os.path.join(out_path, ".".join([language, 'v4_conll']))
    with open(conll, 'w') as CoNLL:
        for i in tqdm(range(len(data['doc_id']))):
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
                                                                                       data["coref"][i]))
            elif i == len(data['doc_id']) - 1:
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
                                                                                       data["coref"][i]))
                CoNLL.write('\n')
                CoNLL.write('#end document\n')
            else:
                if data['doc_id'][i] == data['doc_id'][i + 1]:
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
                                                                                           data["coref"][i]))
                    if data["word_number"][i + 1] == 0:
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
                                                                                           data["coref"][i]))
                    CoNLL.write('\n')
                    CoNLL.write('#end document\n')
                    CoNLL.write('#begin document ({}); part {}\n'.format(data['doc_id'][i + 1], data["part_id"][i + 1]))

    print('End of convertion. Time - {}'.format(time.time() - start))
    return None


def split_doc(inpath, outpath, language='russian'):
    """
    It splits one large conll file containing the entire RuCorp dataset into many separate documents.
    Args:
        inpath: -
        outpath: -
        language: -

    Returns: Nothing

    """
    # split massive conll file to many little
    
    print('Start of splitting ...')
    with open(inpath, 'r+') as f:
        lines = f.readlines()
        f.close()
    set_ends = []
    k = 0
    print('Splitting conll document ...')
    for i in range(len(lines)):
        if lines[i].startswith('#begin'):
            doc_num = lines[i].split(' ')[2][1:-2]
        elif lines[i] == '#end document\n':
            set_ends.append([k, i, doc_num])
            k = i + 1
    for i in range(len(set_ends)):
        cpath = os.path.join(outpath, ".".join([str(set_ends[i][2]), language, 'v4_conll']))
        with open(cpath, 'w') as c:
            for j in range(set_ends[i][0], set_ends[i][1] + 1):
                if lines[j] == '#end document\n':
                    c.write(lines[j][:-1])
                else:
                    c.write(lines[j])
            c.close()

    del lines
    print('Splitts {} docs in {}.'.format(len(set_ends), outpath))
    del set_ends
    del k

    return None


def train_test_split(inpath, train, test, split, random_seed):
    """
    RuCor doesn't provide train/test data splitting, it makes random splitting.
    Args:
        inpath: path to data
        train: path to train folder
        test: path to test folder
        split: int, split ratio
        random_seed: seed for random module

    Returns:

    """
    print('Start train-test splitting ...')
    z = os.listdir(inpath)
    doc_split = ShuffleSplit(1, test_size=split, random_state=random_seed)
    for train_indeses, test_indeses in doc_split.split(z): 
        train_set = [z[i] for i in sorted(list(train_indeses))]
        test_set = [z[i] for i in sorted(list(test_indeses))]
    for x in train_set:
        build_data.move(os.path.join(inpath, x), os.path.join(train, x))
    for x in test_set:
        build_data.move(os.path.join(inpath, x), os.path.join(test, x))
    print('End train-test splitts.')
    return None


def get_all_texts_from_tokens_file(tokens_path, out_path):
    """
    Creates file with pure text from RuCorp dataset.
    Args:
        tokens_path: -
        out_path: -

    Returns: Nothing

    """
    lengths = {}
    # determine number of texts and their lengths
    with open(tokens_path, "r") as tokens_file:
        for line in tokens_file:
            doc_id, shift, length, token, lemma, gram = line[:-1].split('\t')
            try:
                doc_id, shift, length = map(int, (doc_id, shift, length))
                lengths[doc_id] = shift + length
            except ValueError:
                pass

    texts = {doc_id: [' '] * length for (doc_id, length) in lengths.items()}
    # read texts
    with open(tokens_path, "r") as tokens_file:
        for line in tokens_file:
            doc_id, shift, length, token, lemma, gram = line[:-1].split('\t')
            try:
                doc_id, shift, length = map(int, (doc_id, shift, length))
                texts[doc_id][shift:shift + length] = token
            except ValueError:
                pass
    for doc_id in texts:
        texts[doc_id] = "".join(texts[doc_id])

    with open(out_path, "w") as out_file:
        for doc_id in texts:
            out_file.write(texts[doc_id])
            out_file.write("\n")
    return None


def get_char_vocab(input_filename, output_filename):
    """
    Gets chars dictionary from text, and write it into output_filename.
    Args:
        input_filename: -
        output_filename: -

    Returns: Nothing

    """
    data = open(input_filename, "r").read()
    vocab = sorted(list(set(data)))

    with open(output_filename, 'w') as f:
        for c in vocab:
            f.write(u"{}\n".format(c))
    print("[Wrote {} characters to {}] ...".format(len(vocab), output_filename))


def conll2dict(conll, iter_id=None, agent=None, mode='train', doc=None, epoch_done=False):
    """
    Opens the document, reads it, and adds its contents as a string to the dictionary.
    Args:
        conll: path to the conll file
        iter_id: number of operations
        agent: agent name
        mode: train/valid mode
        doc: document name
        epoch_done: flag

    Returns: dict { 'iter_id': iter_id,
                    'id': agent,
                    'epoch_done': epoch_done,
                    'mode': mode,
                    'doc_name': doc
                    'conll_str': s}

    """
    data = {'iter_id': iter_id,
            'id': agent,
            'epoch_done': epoch_done,
            'mode': mode,
            'doc_name': doc}

    with open(conll, 'r', encoding='utf8') as f:
        s = f.read()
        data['conll_str'] = s
        f.close()
    return data


def dict2conll(data, predict):
    """Writes conll string in file"""
    with open(predict, 'w') as CoNLL:
        CoNLL.write(data['conll_str'])
        CoNLL.close()
    return None


def make_summary(value_dict):
    """Make tf.Summary for tensorboard"""
    return tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=v) for k, v in value_dict.items()])


def summary(value_dict, global_step, writer):
    """Make tf.Summary for tensorboard"""
    summary = tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=v) for k, v in value_dict.items()])
    writer.add_summary(summary, global_step)
    return None


def extract_data(infile):
    """
    Extract useful information from conll file and write it in special dict structure.

    {'doc_name': name,
     'parts': {'0': {'text': text,
                    'chains': {'1421': {'chain_id': 'bfdb6d6f-6529-46af-bd39-cf65b96479b5',
                                        'mentions': [{'sentence_id': 0,
                                                    'start_id': 0,
                                                    'end_id': 4,
                                                    'start_line_id': 1,
                                                    'end_line_id': 5,
                                                    'mention': 'Отличный отель для бюджетного отдыха',
                                                    'POS': ['Afpmsnf', 'Ncmsnn', 'Sp-g', 'Afpmsgf', 'Ncmsgn'],
                                                    'mention_id': 'f617f5ca-ed93-49aa-8eae-21b672c5a9df',
                                                    'mention_index': 0}
                                        , ... }}}}}

    Args:
        infile: path to ground conll file

    Returns:
        dict with an alternative mention structure
    """

    # print('Processing: ', basename(infile))
    # filename, parts: [{text: [sentence, ...], chains}, ...]
    data = {'doc_name': None, 'parts': dict()}
    with open(infile, 'r', encoding='utf8') as fin:
        current_sentence = []
        current_sentence_pos = []
        sentence_id = 0
        opened_labels = dict()
        # chain: chain_id -> list of mentions
        # mention: sent_id, start_id, end_id, mention_text, mention_pos
        chains = dict()
        part_id = None
        mention_index = 0
        for line_id, line in enumerate(fin.readlines()):
            # if end document part then add chains to data
            if '#end document' == line.strip():
                assert part_id is not None, print('line_id: ', line_id)
                data['parts'][part_id]['chains'] = chains
                chains = dict()
                sentence_id = 0
                mention_index = 0
            # start or end of part
            if '#' == line[0]:
                continue
            line = line.strip()
            # empty line between sentences
            if len(line) == 0:
                assert len(current_sentence) != 0
                assert part_id is not None
                if part_id not in data['parts']:
                    data['parts'][part_id] = {'text': [], 'chains': dict()}
                data['parts'][part_id]['text'].append(' '.join(current_sentence))
                current_sentence = []
                current_sentence_pos = []
                sentence_id += 1
            else:
                # line with data
                line = line.split()
                assert len(line) >= 3, print('assertion:', line_id, len(line), line)
                current_sentence.append(line[3])
                doc_name, part_id, word_id, word, ref = line[0], int(line[1]), int(line[2]), line[3], line[-1]
                pos, = line[4],
                current_sentence_pos.append(pos)

                if data['doc_name'] is None:
                    data['doc_name'] = doc_name

                if ref != '-':
                    # we have labels like (322|(32)
                    ref = ref.split('|')
                    # get all opened references and set them to current word
                    for i in range(len(ref)):
                        ref_id = int(ref[i].strip('()'))
                        if ref[i][0] == '(':
                            if ref_id in opened_labels:
                                opened_labels[ref_id].append((word_id, line_id))
                            else:
                                opened_labels[ref_id] = [(word_id, line_id)]

                        if ref[i][-1] == ')':
                            assert ref_id in opened_labels
                            start_id, start_line_id = opened_labels[ref_id].pop()

                            if ref_id not in chains:
                                chains[ref_id] = {
                                    'chain_id': str(uuid.uuid4()),
                                    'mentions': [],
                                }

                            chains[ref_id]['mentions'].append({
                                'sentence_id': sentence_id,
                                'start_id': start_id,
                                'end_id': word_id,
                                'start_line_id': start_line_id,
                                'end_line_id': line_id,
                                'mention': ' '.join(current_sentence[start_id:word_id + 1]),
                                'POS': current_sentence_pos[start_id:word_id + 1],
                                'mention_id': str(uuid.uuid4()),
                                'mention_index': mention_index,
                            })
                            assert len(current_sentence[start_id:word_id+1]) > 0, (doc_name, sentence_id, start_id,
                                                                                   word_id, current_sentence)
                            mention_index += 1
    return data


def ana_doc_score(g_file, p_file):
    """Anaphora scores from gold and predicted files.
    A weak criterion for antecedent identification is used.

    Args:
        g_file: path to ground truth conll file
        p_file: path to predicted conll file

    Returns:
        precision, recall, F1-metrics
    """

    main = {}

    g_data = extract_data(g_file)
    p_data = extract_data(p_file)
    pnames = list(p_data['parts'][0]['chains'].keys())
    gnames = list(g_data['parts'][0]['chains'].keys())

    if len(pnames) != len(gnames):
        print('In documents, a different number of clusters.'
              ' {0} in gold file and {1} in predicted file.'.format(len(gnames), len(pnames)))
    else:
        print('The documents have the same number of clusters.')

    for x in [p_data, g_data]:
        for y in x['parts'][0]['chains'].keys():
            for z in x['parts'][0]['chains'][y]['mentions']:
                if not z['start_line_id'] != z['end_line_id'] and z['POS'][0].startswith('P'):
                    c = (z['start_line_id'], z['end_line_id'])
                    if (z['start_line_id'], z['end_line_id']) not in main:
                        main[c] = dict()
                        main[c]['pred'] = list()
                        main[c]['gold'] = list()
                        if x is g_data:
                            main[c]['g_clus'] = gnames.index(y)
                            main[c]['g_clus_real'] = y
                        else:
                            main[c]['p_clus'] = pnames.index(y)
                            main[c]['p_clus_real'] = y
                    else:
                        if x is g_data:
                            main[c]['g_clus'] = gnames.index(y)
                            main[c]['g_clus_real'] = y
                        else:
                            main[c]['p_clus'] = pnames.index(y)
                            main[c]['p_clus_real'] = y

    for x in main:
        for y in g_data['parts'][0]['chains'][main[x]['g_clus_real']]['mentions']:
            if y['start_line_id'] < x[0] and y['end_line_id'] < x[1]:
                main[x]['gold'].append((y['start_line_id'], y['end_line_id']))
            elif y['start_line_id'] < x[0] and y['end_line_id'] == x[1]:
                main[x]['gold'].append((y['start_line_id'], y['end_line_id']))

        for y in p_data['parts'][0]['chains'][main[x]['p_clus_real']]['mentions']:
            if y['start_line_id'] < x[0] and y['end_line_id'] < x[1]:
                main[x]['pred'].append((y['start_line_id'], y['end_line_id']))
            elif y['start_line_id'] < x[0] and y['end_line_id'] == x[1]:
                main[x]['pred'].append((y['start_line_id'], y['end_line_id']))

    # compute F1 score
    score = 0
    fn = 0
    fp = 0

    for x in main:
        k = 0
        if len(main[x]['pred']) != 0 and len(main[x]['gold']) != 0:
            for y in main[x]['pred']:
                if y in main[x]['gold']:
                    score += 1
                    k += 1
                    continue
            if k == 0:
                if main[x]['g_clus'] == main[x]['p_clus']:
                    fn += 1
                    # print('fn', fn)
                    # print(main[x])
                else:
                    fp += 1
                    # print('fp', fp)
                    # print(main[x])
        else:
            continue

    precision = score / (score + fp)
    recall = score / (score + fn)
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def true_ana_score(g_file, p_file):
    main = {}

    g_data = extract_data(g_file)
    p_data = extract_data(p_file)
    pnames = list(p_data['parts'][0]['chains'].keys())
    gnames = list(g_data['parts'][0]['chains'].keys())

    if len(pnames) != len(gnames):
        print('In documents, a different number of clusters.'
              ' {0} in gold file and {1} in predicted file.'.format(len(gnames), len(pnames)))
    else:
        print('The documents have the same number of clusters.')

    for x in [p_data, g_data]:
        for y in x['parts'][0]['chains'].keys():
            for z in x['parts'][0]['chains'][y]['mentions']:
                if not z['start_line_id'] != z['end_line_id'] and z['POS'][0].startswith('P'):
                    c = (z['start_line_id'], z['end_line_id'])
                    if (z['start_line_id'], z['end_line_id']) not in main:
                        main[c] = dict()
                        main[c]['pred'] = list()
                        main[c]['gold'] = list()
                        if x is g_data:
                            main[c]['g_clus'] = gnames.index(y)
                            main[c]['g_clus_real'] = y
                        else:
                            main[c]['p_clus'] = pnames.index(y)
                            main[c]['p_clus_real'] = y
                    else:
                        if x is g_data:
                            main[c]['g_clus'] = gnames.index(y)
                            main[c]['g_clus_real'] = y
                        else:
                            main[c]['p_clus'] = pnames.index(y)
                            main[c]['p_clus_real'] = y

    for x in main:
        for y in g_data['parts'][0]['chains'][main[x]['g_clus_real']]['mentions']:
            if y['start_line_id'] < x[0] and y['end_line_id'] < x[1]:
                main[x]['gold'].append((y['start_line_id'], y['end_line_id']))
            elif y['start_line_id'] < x[0] and y['end_line_id'] == x[1]:
                main[x]['gold'].append((y['start_line_id'], y['end_line_id']))

        for y in p_data['parts'][0]['chains'][main[x]['p_clus_real']]['mentions']:
            if y['start_line_id'] < x[0] and y['end_line_id'] < x[1]:
                main[x]['pred'].append((y['start_line_id'], y['end_line_id']))
            elif y['start_line_id'] < x[0] and y['end_line_id'] == x[1]:
                main[x]['pred'].append((y['start_line_id'], y['end_line_id']))

    # compute F1 score
    tp = 0
    fn = 0
    fp = 0
    prec = list()
    rec = list()
    f_1 = list()

    for x in main:
        if len(main[x]['pred']) != 0 and len(main[x]['gold']) != 0:
            for y in main[x]['pred']:
                if y in main[x]['gold']:
                    tp += 1
                else:
                    fp += 1
            for y in main[x]['gold']:
                if y not in main[x]['pred']:
                    fn += 1

        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f = 2 * p * r / (p + r)
        prec.append(p)
        rec.append(r)
        f_1.append(f)

    precision = np.mean(prec)
    recall = np.mean(rec)
    f1 = np.mean(f_1)

    return precision, recall, f1


def anaphora_score(keys_path, predicts_path):
    """Anaphora scores predicted files.
    A weak criterion for antecedent identification is used.

    Args:
        keys_path: path to ground truth conll files
        predicts_path: path to predicted conll files

    Returns:
        dict with scores
    """

    key_files = list(filter(lambda x: x.endswith('conll'), os.listdir(keys_path)))
    pred_files = list(filter(lambda x: x.endswith('conll'), os.listdir(predicts_path)))

    # assert len(key_files) == len(pred_files), ('The number of visible files is not equal.')

    results = dict()
    results['precision'] = list()
    results['recall'] = list()
    results['F1'] = list()

    result = dict()

    for file in tqdm(pred_files):
        predict_file = os.path.join(predicts_path, file)
        gold_file = os.path.join(keys_path, file)
        p, r, f1 = true_ana_score(gold_file, predict_file)
        results['precision'].append(p)
        results['recall'].append(r)
        results['F1'].append(f1)

    result['precision'] = np.mean(results['precision'])
    result['recall'] = np.mean(results['recall'])
    result['F1'] = np.mean(results['F1'])

    return result
