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

import copy
from parlai.core.agents import Agent
from . import config
from .models import CorefModel
from . import utils
import collections
from collections import Counter

class DocumentState(object):
    def __init__(self):
        self.doc_key = None
        self.text = []
        self.text_speakers = []
        self.speakers = []
        self.sentences = []
        self.clusters = collections.defaultdict(list)
        self.stacks = collections.defaultdict(list)

    def assert_empty(self):
        assert self.doc_key is None
        assert len(self.text) == 0
        assert len(self.text_speakers) == 0
        assert len(self.sentences) == 0
        assert len(self.speakers) == 0
        assert len(self.clusters) == 0
        assert len(self.stacks) == 0

    def assert_finalizable(self):
        assert self.doc_key is not None
        assert len(self.text) == 0
        assert len(self.text_speakers) == 0
        assert len(self.sentences) > 0
        assert len(self.speakers) > 0
        assert all(len(s) == 0 for s in self.stacks.values())

    def finalize(self):
        merged_clusters = []
        for c1 in self.clusters.values():
            existing = None
            for m in c1:
                for c2 in merged_clusters:
                    if m in c2:
                        existing = c2
                        break
                if existing is not None:
                    break
            if existing is not None:
                print("Merging clusters (shouldn't happen very often.)")
                existing.update(c1)
            else:
                merged_clusters.append(set(c1))
        merged_clusters = [list(c) for c in merged_clusters]
        all_mentions = utils.flatten(merged_clusters)
        # print len(all_mentions), len(set(all_mentions))

        if len(all_mentions) != len(set(all_mentions)):
            c = Counter(all_mentions)
            for x in c:
                if c[x] > 1:
                    z = x
                    break
            for i in range(len(all_mentions)):
                if all_mentions[i] == z:
                    all_mentions.remove(all_mentions[i])
                    break
        assert len(all_mentions) == len(set(all_mentions))

        return {
            "doc_key": self.doc_key,
            "sentences": self.sentences,
            "speakers": self.speakers,
            "clusters": merged_clusters
        }

def normalize_word(word):
    if word == "/." or word == "/?":
        return word[1:]
    else:
        return word

def conll2modeldata(data):
    document_state = DocumentState()
    document_state.assert_empty()
    document_state.doc_key = "{}_{}".format(data['doc_id'][0], int(data['part_id'][0]))
    for i in range(len(data['doc_id'])):
        word = normalize_word(data['word'][i])
        coref = data['coreference'][i]
        speaker = data['speaker'][i]
        word_index = i + 1
        document_state.text.append(word)
        document_state.text_speakers.append(speaker)

        if data['part_of_speech'][i] == 'SENT':
            document_state.sentences.append(tuple(document_state.text))
            del document_state.text[:]
            document_state.speakers.append(tuple(document_state.text_speakers))
            del document_state.text_speakers[:]
            pass
        else:
            if coref == "-":
                pass
            else:
                for segment in coref.split("|"):
                    if segment[0] == "(":
                        if segment[-1] == ")":
                            cluster_id = int(segment[1:-1])
                            document_state.clusters[cluster_id].append((word_index, word_index))
                        else:
                            cluster_id = int(segment[1:])
                            document_state.stacks[cluster_id].append(word_index)
                    else:
                        cluster_id = int(segment[:-1])
                        start = document_state.stacks[cluster_id].pop()
                        document_state.clusters[cluster_id].append((start, word_index))

    document_state.assert_finalizable()
    return document_state.finalize()

class CoreferenceAgent(Agent):

    @staticmethod
    def add_cmdline_args(argparser):
        config.add_cmdline_args(argparser)

    def __init__(self, opt, shared=None):
        self.id = 'Coreference_Agent'
        self.episode_done = True
        super().__init__(opt, shared)

        if shared is not None:
            self.is_shared = True
            return

        # Set up params/logging/dicts
        self.is_shared = False
        self.model = CorefModel(opt)

    def observe(self, observation):
        self.observation = copy.deepcopy(observation)
        self.obs_dict = conll2modeldata(self.observation)
        return self.obs_dict

    def act(self):
        return self.batch_act([self.obs_dict])

    def batch_act(self, observations):
        if self.is_shared:
            raise RuntimeError("Parallel act is not supported.")

        tf_loss = self.model.train_op(observations)

        report = {}
        report['step'] = observations['iter_id']
        report['Loss'] = tf_loss
        return report


    # def eval(self, x, y):
    #     loss = self.sess.run(self.loss, feed_dict={self.x: x, self.y_ground_truth: y})
    #     return loss

    # def predict(self, x, xc):
    #     y = self.sess.run(self.y_predicted, feed_dict={self.x: x, self.xc: xc})
    #     return y

    def save(self):
        self.model.save()

    def load(self):
        self.model.init_from_saved()

    def shutdown(self):
        if not self.is_shared:
            if self.model is not None:
                self.model.shutdown()
            self.model = None
