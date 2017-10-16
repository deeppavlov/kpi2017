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



import fasttext
import numpy as np
import os
import tensorflow as tf
from multiprocessing import Pool
from parlai.core.agents import Agent
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from . import utils
from .model import MentionScorerModel
from ...utils import coreference_utils

class EchoAgent(Agent):

    def __init__(self, opt):
        self.observation = None
        self.id = 'EchoAgent'

    def act(self):
        if self.observation is None:
            return {'text': 'Nothing to answer yet.'}
        return self.observation

    def observe(self, observation):
        self.observation = observation
        print(observation.keys())

class CoreferenceAgent(Agent):

    @staticmethod
    def add_cmdline_args(argparser):
        group = argparser.add_argument_group('Coreference Agent')
        
        group.add_argument('--batch_size', type=int, default=128, help='batch size')
        group.add_argument('--save_model_every', type=int, default=10000, help='save model every X iterations')
        group.add_argument('--inner_epochs', type=int, default=300, help='how many times to learn on full observation')
        group.add_argument('--dense_hidden_size', type=int, default=256, help='dense hidden size')
        group.add_argument('--keep_prob_input', type=float, default=0.5, help='dropout keep_prob parameter on inputs')
        group.add_argument('--keep_prob_dense', type=float, default=0.8, help='dropout keep_prob parameter between layers')
        group.add_argument('--lr', type=float, default=0.0005, help='learning rate')
        group.add_argument('--threshold_steps', type=int, default=50, help='how many steps in threshold selection')
        group.add_argument('--print_train_loss_every', type=int, default=500, help='print train loss every X iterations')
        group.add_argument('--print_test_loss_every', type=int, default=500, help='print test loss every X iterations')

        group.add_argument('--scorer_n_threads', type=int, default=25, help='how many threads can CoNLL scorer use')

        group.add_argument('--linkage_method', type=str, default='centroid', help='linkage method: single, complete, averaged, ... look at SciPy doc')

        group.add_argument('--tmp_folder', type=str, default='tmp',
                           help='folder where to dump conll predictions, scorer will use this folder')
        group.add_argument('--embeddings_url', type=str, default='http://share.ipavlov.mipt.ru:8080/repository/embeddings/ft_0.8.3_nltk_yalen_sg_300.bin')      
        group.add_argument('--embeddings_filename', type=str, default='fasttext_embdgs.bin')
        
        group.add_argument('--tensorboard', type=str, default='tensorboard_coreference_scorer', help='path to tensorboard logs')


    def __init__(self, opt):
        print(opt)
        self.opt = opt
        self.last_observation = None
        self.id = 'CoreferenceAgent'
        self.embeddings_url = opt['embeddings_url']
        
        self.agent_dir = opt['model_file']
        self.datapath = os.path.join(opt['datapath'], 'coreference', opt['language'])
        self.embeddings_path = os.path.join(self.agent_dir, opt['embeddings_filename'])
        self.tensorboard_path = os.path.join(self.agent_dir, opt['tensorboard'])

        if not os.path.isdir(self.tensorboard_path):
            os.makedirs(self.tensorboard_path)

        self.tmp_folder = os.path.join(self.agent_dir, opt['tmp_folder'])

        if not os.path.isdir(self.tmp_folder):
            os.makedirs(self.tmp_folder)

        self.valid_path = os.path.join(self.datapath, 'valid')

        self.scorer_path = os.path.join(self.datapath, opt['scorer_path'])

        self.batch_size = opt['batch_size']
        self.inner_epochs = opt['inner_epochs']
        
        self.linkage_method = opt['linkage_method']
        self.threshold_steps = opt['threshold_steps']

        utils.download_embeddings(self.embeddings_url, self.embeddings_path)

        self.embeddings = fasttext.load_model(self.embeddings_path)

        self.data = None
        self.data_valid = None

        # create model and batch_generator on first observe call
        self.model = None
        self.session = None
        self.tf_config = tf.ConfigProto()
        self.tf_config.gpu_options.allow_growth=True
        self.data_bg = None
        self.valid_bg = None
        
        self.best_threshold = 0
        self.best_conll_f1 = 0

        self.global_step = 0

        self.n_threads = opt['scorer_n_threads']
        assert self.n_threads > 0 or self.n_threads == -1

        run_number = len(os.listdir(self.tensorboard_path))
        self.run_path = os.path.join(self.tensorboard_path, 'run_{0}_scorer'.format(run_number))
        print('run_path: {}'.format(self.run_path))


    def act(self):
        if self.last_observation is None:
            return {'text': 'Nothing to answer yet.'}

        # if data is not None -> train model on data and make prediction
        # if data is None -> make prediction

        if self.data is not None and len(self.data) > 0:
            self.best_conll_f1 = 0
            
            self._train_scorer()
            print('Making scorer predictions on valid dataset...')
            predicted_scores = self._get_scorer_predictions(self.valid_bg)
            clustering, min_score, max_score = utils.build_clusters(predicted_scores, method=self.linkage_method)
            
            print('Making conll predictions on valid dataset...')
            pool_args = []
            for t in tqdm(np.linspace(min_score, max_score, self.threshold_steps)):
                output_path = '{}_{:.2f}'.format(os.path.join(self.tmp_folder, 'output'), t)
                doc_to_chains = utils.make_dendrogram_predictions(self.valid_bg.dl, clustering, threshold=t)
                pool_args.append((self.scorer_path, self.valid_path, output_path))
                for doc in doc_to_chains:
                    utils.make_prediction_file(self.data_valid_conll[doc], self.data_valid[doc], output_path, doc_to_chains[doc])

            print('Scoring all conll predicions on valid dataset')

            results = None
            with Pool(self.n_threads) as pool:
                results = pool.starmap(coreference_utils.score, pool_args)
            
            assert results is not None
            for t, res in zip(np.linspace(min_score, max_score, self.threshold_steps), results):
                if self.best_conll_f1 < res['conll-F-1']:
                    self.best_conll_f1 = res['conll-F-1']
                    self.best_threshold = t

            # clean tmp dir
            for root, dirs, files in os.walk(self.tmp_folder, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))

        print('Making scorer predictions on dataset...')
        predicted_scores = self._get_scorer_predictions(self.valid_bg)
        clustering, min_score, max_score = utils.build_clusters(predicted_scores, method=self.linkage_method)
        print('Making conll predictions on dataset with threshold: {:.2f} and conll-F-1: {:.3f} on valid'.format(self.best_threshold, self.best_conll_f1))
        doc_to_chains = utils.make_dendrogram_predictions(self.valid_bg.dl, clustering, threshold=self.best_threshold)
        pred_conll_lines = [utils.make_prediction_file(self.data_valid_conll[doc], self.data_valid[doc], None, doc_to_chains[doc], write=False) for doc in doc_to_chains]
        self.data_bg, self.valid_bg = None, None
        return {'id': self.id, 'valid_conll': pred_conll_lines}        

    def observe(self, observation):
        self.last_observation = observation
        # extract data from conll
        print('Preprocessing observation')
        self.data = [utils.extract_data_from_conll(conll_lines) for conll_lines in self.last_observation['conll']]
        self.data_conll = {extracted['doc_name']: conll for extracted, conll in zip(self.data, self.last_observation['conll'])}
        self.data = {el['doc_name']:el for el in self.data}
        self.data_valid = [utils.extract_data_from_conll(conll_lines) for conll_lines in self.last_observation['valid_conll']]
        self.data_valid_conll = {extracted['doc_name']: conll for extracted, conll in zip(self.data_valid, self.last_observation['valid_conll'])}
        self.data_valid = {el['doc_name']:el for el in self.data_valid}
        # extract simple features
        self.data_smpl = {doc: utils.generate_simple_features(self.data[doc]) for doc in self.data}
        self.data_valid_smpl = {doc: utils.generate_simple_features(self.data_valid[doc]) for doc in self.data_valid}
        # extract embeddings features
        self.data_emb = {doc: utils.generate_emb_features(self.data[doc], self.embeddings) for doc in self.data}
        self.data_valid_emb = {doc: utils.generate_emb_features(self.data_valid[doc], self.embeddings) for doc in self.data_valid}
        print('Preprocessing finished')
        # create batch generators
        if self.data is not None and len(self.data) > 0:
            self.data_bg = utils.MentionPairsBatchGenerator(self.data, self.data_emb, self.data_smpl)
        else:
            self.data_bg = None
        
        if self.data_valid is not None and len(self.data_valid) > 0:
            self.valid_bg = utils.MentionPairsBatchGenerator(self.data_valid, self.data_valid_emb, self.data_valid_smpl)

        # create model
        if self.model is None:
            self.model = MentionScorerModel(hidden_size=self.opt['dense_hidden_size'], lr=self.opt['lr'],
                keep_prob_input=self.opt['keep_prob_input'], keep_prob_dense=self.opt['keep_prob_dense'], features_size=self.valid_bg.dl.features_size)
            self.session = tf.Session(config=self.tf_config)
            tf.global_variables_initializer().run(session=self.session)
            if 'pretrained_model' in self.opt:
                checkpoint = tf.train.latest_checkpoint(self.opt['pretrained_model'])
                print('Initializing model from checkpoint: {}'.format(checkpoint))
                saver = tf.train.Saver()
                #print('Loading from:', checkpoint)
                saver.restore(self.session, checkpoint)
                with open(os.path.join(self.agent_dir, 'threshold'), 'r') as fin:
                    self.best_threshold = float(fin.readline().strip())
                    self.best_conll_f1 = float(fin.readline().strip().split()[-1])

    
    def save(self):
        if self.session is not None:
            saver = tf.train.Saver()
            saver.save(self.session, os.path.join(self.agent_dir, 'model'))
            with open(os.path.join(self.agent_dir, 'threshold'), 'w') as fout:
                fout.write('{}\n'.format(self.best_threshold))
                fout.write('conll-f-1: {:.5f}\n'.format(self.best_conll_f1))

    def shutdown(self):
        tf.reset_default_graph()


    def _train_scorer(self):
        summary_writer = tf.summary.FileWriter(self.run_path, graph=self.session.graph)
        saver = tf.train.Saver(max_to_keep=None)

        while self.data_bg.epoch < self.inner_epochs:
            A, A_f, B, B_f, AB_f, C = self.data_bg.get_batch(self.batch_size)

            loss, loss_sum, logits = self.model.train_batch(self.session, A, A_f, B, B_f, AB_f, C)

            summary_writer.add_summary(loss_sum, self.global_step)
            
            if self.global_step % 10 == 0:
                summary_writer.flush()

            logits = np.argmax(logits, axis=1)
            
            if self.global_step % self.opt['print_train_loss_every'] == 0:
                print('TRAIN: iter: {} loss: {}'.format(self.global_step, loss))
                print('TRAIN:\ttarget and predict:')
                print('Y:', C[:20])
                print('P:', logits[:20])
            
            if self.global_step % self.opt['print_test_loss_every'] == 0:
                A, A_f, B, B_f, AB_f, C = self.valid_bg.get_batch(batch_size=self.batch_size)
                loss, loss_test_sum, logits, pred = self.model.test_batch(self.session, A, A_f, B, B_f, AB_f, C)
                summary_writer.add_summary(loss_test_sum, self.global_step)
                logits = np.argmax(logits, axis=1)
                roc_auc = roc_auc_score(C, pred[:,1])
                [roc_auc_sum] = self.session.run([self.model.roc_auc_summary], feed_dict={self.model.roc_auc: roc_auc})
                summary_writer.add_summary(roc_auc_sum, self.global_step)
                print('TEST: iter: {} loss: {} roc_auc: {}'.format(self.global_step, loss, roc_auc))
                print('TEST:\ttarget and predict:')
                print('Y:', C[:20])
                print('P:', logits[:20])

            
            if self.global_step % self.opt['save_model_every'] == 0:
                saver.save(self.session, os.path.join(self.run_path, 'model') , global_step=self.global_step)
                pass
            self.global_step += 1

    def _get_scorer_predictions(self, bg):
        '''
        bg: batch generator to use to make predictions; get_document_batch is called
        '''
        predicted_scores = []
        for doc_id in tqdm(range(bg.max_doc_id)):
            A, A_f, B, B_f, AB_f = bg.get_document_batch(doc_id)
            if A is None:
                predicted_scores.append([])
                continue
            # A, B can be very large so lets split them on batches
            A_b = utils.split_on_batches(A, self.batch_size)
            A_f_b = utils.split_on_batches(A_f, self.batch_size)
            B_b = utils.split_on_batches(B, self.batch_size)
            B_f_b = utils.split_on_batches(B_f, self.batch_size)
            AB_f_b = utils.split_on_batches(AB_f, self.batch_size)
            batch_pred = []
            for a, a_f, b, b_f, ab_f in zip(A_b, A_f_b, B_b, B_f_b, AB_f_b):
                [pred] = self.session.run([self.model.pred], feed_dict={
                    self.model.A: a,
                    self.model.A_features: a_f,
                    self.model.B: b,
                    self.model.B_features: b_f,
                    self.model.AB_features: ab_f,
                    self.model.keep_prob_input_ph: 1.0,
                    self.model.keep_prob_dense_ph: 1.0
                    })
                batch_pred.append(pred[:,0])
            predicted_scores.append(np.reshape(np.concatenate(batch_pred), 
                (int(np.sqrt(len(A))), int(np.sqrt(len(A))))))
        return predicted_scores
