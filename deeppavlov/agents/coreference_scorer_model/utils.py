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


import os
import random
import urllib
import uuid

import numpy as np
from scipy.cluster.hierarchy import linkage
from tqdm import tqdm


def download_embeddings(url, embeddings_path):
    """downloads embeddings from url and puts to embeddings_path"""
    print('Loading embeddings: {}'.format(embeddings_path))
    if not os.path.isfile(embeddings_path):
        print('There is no such file: {}\nDownloading from {} ...'.format(embeddings_path, url))
        try:
            urllib.request.urlretrieve(url, embeddings_path)
            print('Successfully downloaded to {}'.format(embeddings_path))
        except Exception as e:
            raise RuntimeError('Failed to download from: {}'.format(url))


def extract_data_from_conll(conll_lines):
    """

    Args:
        conll_lines: list of lines in conll file (result of readlines())

    Returns:
        dict representation of conll file; For example:
        {
        "doc_name": "wb/a2e/00/a2e_0000",
        "parts": {
            "0": {
                "text": [
                    "Fist line",
                    "Second line",
                ],
                "chains": {
                    "3": {
                        "chain_id": "id",
                        "mentions": [
                            {
                            "sentence_id": 0,
                            "start_id": 1,
                            "end_id": 1,
                            "start_line_id": 0,
                            "end_line_id": 0,
                            "mention": "line",
                            "POS": [
                                "NN"
                            ],
                            "mention_id": "some_id",
                            "mention_index": 0
                            },
                        ]
                    }
                }
        }
    """

    # filename, parts: [{text: [sentence, ...], chains}, ...]
    data = {'doc_name': None, 'parts': dict()}

    current_sentence = []
    current_sentence_pos = []
    sentence_id = 0
    opened_labels = dict()
    # chain: chain_id -> list of mentions
    # mention: sent_id, start_id, end_id, mention_text, mention_pos
    chains = dict()
    part_id = None
    mention_index = 0
    for line_id, line in enumerate(conll_lines):
        line = line.strip()
        # if end document part then add chains to data
        if '#end document' == line:
            assert part_id is not None, print('line_id: ', line_id)
            data['parts'][part_id]['chains'] = chains
            chains = dict()
            sentence_id = 0
            mention_index = 0
        # start or end of part  
        if len(line) > 0 and '#' == line[0]:
            continue
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
            line = line.split('\t')
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
                    ref_id = ref[i].strip('()')
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
                        mention_index += 1
    return data


def generate_simple_features(data):
    """method for generating non-embeddings features

    Args:
        data: dict representation of conll file
    """
    features = dict()
    for part_id, part in data['parts'].items():
        mentions_count = sum(map(lambda x: len(x), part['chains'].values()))
        tokens_count = sum(map(lambda x: len(x.split()), part['text']))
        for chain_id in sorted(part['chains'].keys()):
            chain = part['chains'][chain_id]
            for mention in chain['mentions']:
                sentence_id = mention['sentence_id']
                start_id = mention['start_id']
                end_id = mention['end_id']
                mention_id = mention['mention_id']
                mention_index = mention['mention_index']
                POS = mention['POS']

                features[mention_id] = {
                    'mention_index': (mention_index + 1) / (mentions_count + 1),
                    'start_index': (start_id + 1) / (tokens_count + 1),
                    'end_index': (end_id + 1) / (tokens_count + 1),
                    'has_PRP': 1 if 'P' in list(map(lambda x: x[0] if len(x) > 0 else None, POS)) else 0,
                    'has_CD': 1 if 'CD' in POS else 0,
                    'mentions_count': mentions_count + 1,
                    'tokens_count': tokens_count + 1,
                    'mention_index_ohe': distance_to_buckets(mention_index + 1),
                    'mention_start_ohe': distance_to_buckets(start_id + 1),
                    'mention_width_ohe': distance_to_buckets(len(mention['mention'].split())),
                }

    return features


def generate_emb_features(data, ft_model, window_size=5):
    """method for generating embeddings features

    Args:
        data: dict representation of conll file
        ft_model: fasttext embeddings
        window_size: how many neighbors to use

    Returns:
        dict: key: mention_id value: embedding_features
    """
    embeddings_features = dict()
    # TODO get embeddings dim
    zeros = np.zeros_like(ft_model['тест'])
    for part_id, part in data['parts'].items():
        for chain_id in sorted(part['chains'].keys()):
            chain = part['chains'][chain_id]
            for mention in chain['mentions']:
                sentence_id = mention['sentence_id']
                start_id = mention['start_id']
                end_id = mention['end_id']
                mention_id = mention['mention_id']
                embs = np.array([ft_model[word.lower()] / np.linalg.norm(ft_model[word.lower()]) for word in
                                 mention['mention'].split()])
                if len(embs) == 0:
                    # TODO it is a bug in dataset, wrong tokenization and wrong split on sentences
                    embs = [zeros]

                emb_mean = np.mean(embs, axis=0)
                assert len(part['text']) > sentence_id, (data['doc_name'], part_id, sentence_id)
                sentence = part['text'][sentence_id].split()
                prev_embs = np.array(
                    [ft_model[word.lower()] for word in sentence[max(0, start_id - window_size):start_id]])
                next_embs = np.array([ft_model[word.lower()] for word in sentence[end_id + 1:end_id + window_size + 1]])
                prev_embs_mean = np.mean(prev_embs, axis=0) if len(prev_embs) > 0 else zeros
                next_embs_mean = np.mean(next_embs, axis=0) if len(next_embs) > 0 else zeros

                embeddings_features[mention_id] = {
                    'mean': emb_mean,
                    'first': embs[0],
                    'last': embs[-1],
                    'prev_mean': prev_embs_mean if prev_embs.shape[0] > 0 else zeros,
                    'next_mean': next_embs_mean if next_embs.shape[0] > 0 else zeros,
                    'prev_0': prev_embs[-2] if prev_embs.shape[0] > 1 else zeros,
                    'prev_1': prev_embs[-1] if prev_embs.shape[0] > 0 else zeros,
                    'next_0': next_embs[0] if next_embs.shape[0] > 0 else zeros,
                    'next_1': next_embs[1] if next_embs.shape[0] > 1 else zeros,
                }

    embeddings_features = {m: {f: embeddings_features[m][f].tolist() for f in embeddings_features[m]} for m in
                           embeddings_features}
    return embeddings_features


def distance_to_buckets(d):
    """converts distance to one-hot vector"""
    ohe = [0] * 10
    if d == 0:
        ohe[0] = 1
    elif d == 1:
        ohe[1] = 1
    elif d == 2:
        ohe[2] = 1
    elif d == 3:
        ohe[3] = 1
    elif d == 4:
        ohe[4] = 1
    elif 5 <= d < 8:
        ohe[5] = 1
    elif 8 <= d < 16:
        ohe[6] = 1
    elif 16 <= d < 32:
        ohe[7] = 1
    elif 32 <= d < 64:
        ohe[8] = 1
    elif 64 <= d:
        ohe[9] = 1
    assert sum(ohe) == 1, (d, ohe)
    return ohe


class DataLoader:
    """Help class to load and preprocess conll datafiles"""

    def __init__(self, datas, data_embs, data_smpls):
        """DataLoader parameters initialization"""

        # documents [[{chain_id: [mention_id, ...]}, {...}, ...], ...]
        # list of documents
        # document is a dict of chains
        # chain is a list of mention_id
        # {mention_id: mention_features), ...}
        self.datas = datas
        self.data_embs = data_embs
        self.data_smpls = data_smpls

        self.documents = []
        self.document_files = []
        # ordered mentions in document
        # list of ordered lists of mention_ids
        self.document_mentions = []
        self.embeddings = {}
        self.features = {}
        # mention_id -> chain_id
        self.mentions_to_chain = dict()
        # chain_id -> list mention_ids
        self.chain_to_mentions = dict()
        self.chain_to_document = dict()
        # mention_id -> np.array with all features
        self.mention_features = dict()
        self.mentions_sentid = dict()
        self.mentions_pos = dict()
        self.documents_text = []
        self.features_size = 0

        self._load_all_data()

        # pregenerate all feature vectors to increase get_batch speed
        # uses ~4Gb RAM on train set (without pregeneration: ~3Gb but SLOWER!)
        self._generate_all_features()

    def _load_all_data(self):
        # document == part in terms of conll files
        print('DataLoader: loading documents from data')
        for data_key in tqdm(list(sorted(self.datas.keys()))):
            data = self.datas[data_key]
            for part in data['parts'].values():
                chains = {chain['chain_id']: [m['mention_id'] for m in chain['mentions']] for chain in part['chains'].values()}
                mentions = [(m['mention_id'], m['mention_index']) for chain_id in sorted(part['chains'].keys())
                            for m in part['chains'][chain_id]['mentions']]
                mentions_sentid = {m['mention_id']: m['sentence_id'] for chain in part['chains'].values() for m in chain['mentions']}
                mentions_pos = {m['mention_id']: (m['start_id'], m['end_id']) for chain in part['chains'].values() for m in chain['mentions']}
                self.document_mentions.append(list(map(lambda x: x[0], sorted(mentions, key=lambda x: x[1]))))
                self.chain_to_mentions.update(chains)
                self.documents.append(chains)
                self.document_files.append(data['doc_name'])
                self.mentions_sentid.update(mentions_sentid)
                self.mentions_pos.update(mentions_pos)
                self.mentions_to_chain.update({
                    m['mention_id']: chain['chain_id'] for chain in part['chains'].values() for m in chain['mentions']
                })
                self.chain_to_document.update({c: len(self.documents) - 1 for c in chains})
                self.documents_text.append(part['text'])

        print('DataLoader: loading embedding features')
        for data_emb in tqdm(self.data_embs.values()):
            self.embeddings.update(data_emb)

        print('DataLoader: loading other features')
        for data_smpl in tqdm(self.data_smpls.values()):
            self.features.update(data_smpl)

    def _generate_all_features(self):
        """
            generates all features for all mentions
            and frees from memory: self.embeddings and self.features
            
            pregenerate all feature vectors to increase get_batch speed
        """
        print('DataLoader: generating all features')
        # self.mention_features = {m: self._make_mention_features(m) for ms in self.document_mentions for m in ms}
        assert self.embeddings is not None
        assert self.features is not None

        for ms in tqdm(self.document_mentions):
            for m in ms:
                self.mention_features[m] = self._make_mention_features(m)

        self.features_size = len(self.mention_features[m])
        self.embeddings = None
        print('DataLoader: generating all features finished')

    def get_all_mentions_from_doc(self, doc_id):
        return self.document_mentions[doc_id]

    def _make_mention_features(self, m):
        emb = np.vstack([self.embeddings[m][k] for k in sorted(self.embeddings[m].keys())])
        f = np.vstack(
            list([self.features[m][f] for f in ['mention_index', 'start_index', 'end_index', 'has_PRP', 'has_CD']]))
        return np.concatenate((emb.flatten(), f.flatten()))


class MentionPairsBatchGenerator():
    """Object for generation batches of pairs of mentions"""

    def __init__(self, datas, data_embs, data_smpls):
        """MentionPairsBatchGenerator parameters initialization"""
        # batch iterator index
        self.current_doc_id = 0
        self.epoch = 0

        self.dl = DataLoader(datas, data_embs, data_smpls)

        self.max_doc_id = len(self.dl.documents)
        self.features_size = self.dl.features_size

    def _mention_to_features(self, m):
        """building features for single mention"""
        features = self.dl.features[m]
        res = [
            features['has_PRP'],
            features['has_CD'],
            np.argmax(features['mention_index_ohe']),
            np.argmax(features['mention_start_ohe']),
            np.argmax(features['mention_width_ohe'])
        ]
        return res

    def _pair_features(self, A, B):
        """Building features for pair of mentions"""
        AB_f = []
        for a, b in zip(A, B):
            a_f = self.dl.features[a]
            b_f = self.dl.features[b]
            mention_distance = abs(
                int(a_f['mention_index'] * a_f['mentions_count']) - int(b_f['mention_index'] * b_f['mentions_count']))
            words_distance = abs(
                int(a_f['start_index'] * a_f['tokens_count']) - int(b_f['start_index'] * b_f['tokens_count']))
            AB_f.append(
                [np.argmax(distance_to_buckets(mention_distance)), np.argmax(distance_to_buckets(words_distance))])
        return AB_f

    def get_batch(self, batch_size=64):
        """Builds data samples of size batch_size
        score(A, B)
        A: left arguments
        B: right arguments

        Args:
            batch_size: size of batch

        Returns:
            feature representation of mentions and labels

        """
        mentions = []
        while len(mentions) < batch_size * 2:
            mentions.extend(self.dl.get_all_mentions_from_doc(self.current_doc_id))
            self.current_doc_id = (self.current_doc_id + 1) % self.max_doc_id
            if self.current_doc_id == 0:
                self.epoch += 1
        A = random.sample(mentions, batch_size)
        B = []
        for a in A:
            # magic constant to make classes balanced
            if random.random() > 0.53:
                B.append(random.choice(self.dl.chain_to_mentions[self.dl.mentions_to_chain[a]]))
            else:
                B.append(random.choice(mentions))

        labels = [1 if self.dl.mentions_to_chain[x] == self.dl.mentions_to_chain[y] else 0 for x, y in zip(A, B)]
        A_f = [self._mention_to_features(m) for m in A]
        B_f = [self._mention_to_features(m) for m in B]
        AB_f = self._pair_features(A, B)
        A = [self.dl.mention_features[m] for m in A]
        B = [self.dl.mention_features[m] for m in B]
        return np.vstack(A), np.stack(A_f), np.vstack(B), np.stack(B_f), np.stack(AB_f), np.stack(labels)

    def get_document_batch(self, doc_id):
        """builds batch of all mention pairs in one document

        Args:
            doc_id: id of document

        Returns:
            feature representation of mentions and labels
        """
        mentions = self.dl.get_all_mentions_from_doc(doc_id)
        if len(mentions) == 0:
            return None, None
        A, B = [], []
        for a in mentions:
            for b in mentions:
                A.append(a)
                B.append(b)
        A_f = [self._mention_to_features(m) for m in A]
        B_f = [self._mention_to_features(m) for m in B]
        AB_f = self._pair_features(A, B)
        A = [self.dl.mention_features[m] for m in A]
        B = [self.dl.mention_features[m] for m in B]
        return np.vstack(A), np.stack(A_f), np.vstack(B), np.stack(B_f), np.stack(AB_f)

    def reset(self):
        self.current_doc_id = 0
        self.epoch = 0


def make_prediction_file(conll_lines, data, path_to_save, chains, write=True):
    """makes prediction file based on source conll file

    Args:
        conll_lines: source conll file
        data: dict representation of conll file
        path_to_save: where to save predicted conll file
        chains: predicted chains for this document
        write: write output to file or just return list of strings
    Returns:
        list of string == predicted conll file
    """

    lines_to_write = []
    mentions = dict()
    for part_id in data['parts']:
        part = data['parts'][part_id]
        for chain in part['chains'].values():
            for m in chain['mentions']:
                mentions[m['mention_id']] = m

    start_labels = dict()
    end_labels = dict()
    # opens and closes in the same place
    uno_labels = dict()

    chain_id = -1

    # filter chains with only one mention
    chains = [c for p in chains for c in p if len(c) > 1]

    for chain in chains:
        chain_id += 1
        for m_id in chain:
            m = mentions[m_id]
            start = m['start_line_id']
            end = m['end_line_id']

            if start != end:
                if start not in start_labels:
                    start_labels[start] = [chain_id]
                else:
                    start_labels[start].append(chain_id)

                if end not in end_labels:
                    end_labels[end] = [chain_id]
                else:
                    end_labels[end].append(chain_id)
            else:
                if start not in uno_labels:
                    uno_labels[start] = [chain_id]
                else:
                    uno_labels[start].append(chain_id)

    for line_id, line in enumerate(conll_lines):
        line = line.rstrip()
        if line_id not in start_labels and line_id not in end_labels and line_id not in uno_labels:
            if len(line.split()) <= 3 or '#begin document' in line:
                lines_to_write.append(line + '\n')
            else:
                last_space = None
                try:
                    last_space = line.rindex(' ')
                except:
                    last_space = line.rindex('\t')
                lines_to_write.append(line[:last_space + 1] + '-\n')
        else:
            opens = start_labels[line_id] if line_id in start_labels else []
            ends = end_labels[line_id] if line_id in end_labels else []
            unos = uno_labels[line_id] if line_id in uno_labels else []
            s = []
            s += ['({})'.format(el) for el in unos]
            s += ['({}'.format(el) for el in opens]
            s += ['{})'.format(el) for el in ends]
            s = '|'.join(s)
            try:
                last_space = line.rindex(' ')
            except:
                last_space = line.rindex('\t')
            lines_to_write.append(line[:last_space + 1] + '{}\n'.format(s))

    if write:
        if not os.path.isdir(path_to_save):
            os.makedirs(path_to_save)

        # load conll file and change last column
        output_file = os.path.join(path_to_save, data['doc_name'] + '.ru.v4_conll')
        with open(output_file, 'w', encoding='utf8') as fout:
            for line in lines_to_write:
                fout.write(line)

    return lines_to_write


def build_clusters(predicted_scores, method='centroid'):
    """agglomerative clustering using predicted scores as distances

    Args:
        predicted_scores: predicted scores for all mentions in documents
        method: methods for calculating distance between clusters
            look at scipy.cluster.hierarchy.linkage documentation

    Returns:
        clustering, min_score and max_score in predicted_scores

    """
    print('building clusters')
    min_score = 1e10
    max_score = 0
    clustrering = []
    for doc_id in tqdm(range(len(predicted_scores))):
        scores = predicted_scores[doc_id]
        if len(scores) > 0:
            distances = []
            for i in range(len(scores)):
                for j in range(i + 1, len(scores)):
                    distances.append((scores[i, j] + scores[j, i]) / 2)
            c = linkage(distances, method=method)
            clustrering.append(c)
            min_score = min(min(c[:, 2]), min_score)
            max_score = max(max(c[:, 2]), max_score)
    print('clusters are built: min_score: {} max_score: {}'.format(min_score, max_score))
    return clustrering, min_score, max_score


def build_chains(clustering, mentions, threshold=1.0):
    """build coreference chains for one document

    Args:
        clustering: result of build_clusters method
        mentions: mentions of one document
        threshold: min score when to merge two clusters

    Returns:
        coreference chains
    """
    chains = [[m] for m in mentions]
    for i in range(len(clustering)):
        if clustering[i, 2] > threshold:
            break
        chains.append(list(set(chains[int(clustering[i, 0])] + chains[int(clustering[i, 1])])))
        for j in [0, 1]:
            chains[int(clustering[i, j])] = None
    chains = list(filter(lambda x: x is not None, chains))
    return chains


def make_clustering_predictions(dl, clustering, threshold):
    """build coreference chains for all documents in dl

    Args:
        dl: dataloader to use
        clustering: result of build_clusters method
        threshold: min score when to merge two clusters

    Returns:
        coreference chains for documents in dl
    """
    doc_to_chains = dict()
    for doc_id, ms in enumerate(dl.document_mentions):
        # one document
        if clustering[doc_id] is None:
            chains = []
        else:
            chains = build_chains(clustering[doc_id], ms, threshold=threshold)
        doc_name = dl.document_files[doc_id]
        if doc_name in doc_to_chains:
            doc_to_chains[doc_name].append(chains)
        else:
            doc_to_chains[doc_name] = [chains]
    return doc_to_chains


def split_on_batches(data, batch_size):
    """splits array data on batches of size batch_size
    last batch can be of size less then batch_size
    """
    data_batched = []
    for i in range(data.shape[0] // batch_size):
        data_batched.append(data[i * batch_size:(i + 1) * batch_size])
    if data.shape[0] % batch_size > 0:
        data_batched.append(data[(i + 1) * batch_size:])
    return data_batched
