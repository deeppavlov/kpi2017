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

from collections import defaultdict
from os.path import join, basename
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
import numpy as np
import os
import parlai.core.build_data as build_data
import shutil
import tensorflow as tf
import time

def RuCoref2CoNLL(path, out_path, language='ru'):
    data = {"doc_id": [],
            "part_id": [],
            "word_number": [],
            "word": [],
            "part_of_speech": [],
            "parse_bit": [],
            "lemma": [],
            "sense": [],
            "speaker": [],
            "entity": [],
            "predict": [],
            "coref": []}

    part_id = '0'
    speaker = 'spk1'
    sense = '-'
    entity = '-'
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
            doc_id, variant, group_id, chain_id, link, shift, lens, content, tk_shifts, attributes, head, hd_shifts = line[
                                                                                                                      :-1].split('\t')

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
        for line in tokens_file:
            doc_id, shift, length, token, lemma, gram = line[:-1].split('\t')
            if doc_id == 'doc_id':
                continue
            data['word_number'].append(k)
            data['word'].append(token)
            if token == '.':
                k = 0
            else:
                k += 1
            data['doc_id'].append('bc' + doc_id)
            data['part_id'].append(part_id)
            data['lemma'].append(lemma)
            data['part_of_speech'].append(gram[0:-1])
            data['sense'].append(sense)
            data['speaker'].append(speaker)
            data['entity'].append(entity)
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
                                                                                       data["entity"][i],
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
                                                                                       data["entity"][i],
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
                                                                                           data["entity"][i],
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
                                                                                           data["entity"][i],
                                                                                           data["predict"][i],
                                                                                           data["coref"][i]))
                    CoNLL.write('\n')
                    CoNLL.write('#end document\n')
                    CoNLL.write('#begin document ({}); part {}\n'.format(data['doc_id'][i + 1], data["part_id"][i + 1]))

    print('End of convertion: took {0:.3f}s'.format(time.time() - start))
    return None


def get_doc_name(line):
    '''
    returns doc_name from #begin line
    '''
    return line.split('(')[1].split(')')[0]

def split_doc(inpath, outpath, language):
    # split massive conll file to many little
    print('Start of splitting ...')
    with open(inpath, 'r+') as f:
        lines = f.readlines()
        f.close()
    set_ends = []
    doc_names = []
    k = 0
    print('Splitting conll documents ...')
    for i in range(len(lines)):
        if lines[i] == '#end document\n':
            set_ends.append([k, i])
            k = i + 1
        if lines[i][:15] == '#begin document':
            doc_names.append(get_doc_name(lines[i]))
    for i in range(len(set_ends)):
        cpath = os.path.join(outpath, ".".join([doc_names[i], language, 'conll']))
        with open(cpath, 'w') as c:
            for j in range(set_ends[i][0], set_ends[i][1] + 1):
                c.write(lines[j])
            c.close()

    del lines
    print('Splitted {} docs in {}.'.format(len(set_ends), outpath))
    del set_ends
    del k

    return None


def train_valid_test_split(inpath, train_path, valid_path, test_path, valid_ratio, test_ratio, seed=None):
    assert valid_ratio + test_ratio <= 1.0
    assert valid_ratio > 0 and test_ratio > 0
    source_files = os.listdir(inpath)
    
    train_valid, test = train_test_split(source_files, test_size=test_ratio, random_state=seed)
    train, valid = train_test_split(train_valid, test_size=valid_ratio/(1-test_ratio), random_state=seed)
    
    print('train_valid_test_split: {}/{}/{}'.format(len(train), len(valid), len(test)))
    for dataset, data_path in zip([train, valid, test], [train_path, valid_path, test_path]):
        for el in dataset: 
            build_data.move(os.path.join(inpath, el), os.path.join(data_path, el))
    return None


def get_all_texts_from_tokens_file(tokens_path, out_path):
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
    data = open(input_filename, "r").read()
    vocab = sorted(list(set(data)))

    with open(output_filename, 'w') as f:
        for c in vocab:
            f.write(u"{}\n".format(c))
    print("[Wrote {} characters to {}] ...".format(len(vocab), output_filename))

def conll2dict(iter_id, conll, agent, mode, doc, epoch_done=False):
    data = {'doc_id': [],
            'part_id': [],
            'word_number': [],
            'word': [],
            'part_of_speech': [],
            'parse_bit': [],
            'lemma': [],
            'sense': [],
            'speaker': [],
            'entity': [],
            'predict': [],
            'coreference': [],
            'iter_id': iter_id,
            'id': agent,
            'epoch_done': epoch_done,
            'mode': mode,
            'doc_name': doc}

    with open(conll, 'r') as f:
        for line in f:
            row = line.split('\t')
            if row[0].startswith('#'):
                continue
            elif row[0] == '\n':
                data['doc_id'].append('bc')
                data['part_id'].append('0')
                data['word_number'].append('0')
                data['word'].append('.')
                data['part_of_speech'].append('End_of_sentence')
                data['parse_bit'].append('-')
                data['lemma'].append('-')
                data['sense'].append('-')
                data['speaker'].append('spk1')
                data['entity'].append('-')
                data['predict'].append('-')
                data['coreference'].append('-')
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
                data['entity'].append(row[9])
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
                                                    data["entity"][i],
                                                    data["predict"][i],
                                                    data["coreference"][i]))
            elif i == len(data['doc_id'])-1 and data['part_of_speech'][i] == 'End_of_sentence':
                CoNLL.write('#end document\n')
            elif data['part_of_speech'][i] == 'End_of_sentence':
                continue
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
                                                        data["entity"][i],
                                                        data["predict"][i],
                                                        data["coreference"][i]))
                elif data['part_of_speech'][i] == 'End_of_sentence':
                    continue
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
                                                    data["entity"][i],
                                                    data["predict"][i],
                                                    data["coreference"][i]))
                    CoNLL.write('\n')
        CoNLL.close()
    return None

def make_summary(value_dict):
    return tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=v) for k, v in value_dict.items()])

def summary(value_dict, global_step, writer):
    summary = tf.Summary(value=[tf.Summary.Value(tag=k, simple_value=v) for k, v in value_dict.items()])
    writer.add_summary(summary, global_step)
    return None

def save_observations(observation, path_to_save, lang='ru'):
    for lines in observation:
            doc_name = get_doc_name(lines[0])
            with open(os.path.join(path_to_save, doc_name + '.{}.conll'.format(lang)), 'w') as fout:
                for line in lines:
                    fout.write(line)
