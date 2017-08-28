import copy
import os
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from parlai.core.params import class2str
from scipy.io import mmwrite, mmread

from . import config
from .model import InsultsModel
from .utils import create_vectorizer_selector, get_vectorizer_selector, vectorize_select_from_data


class InsultsDictionaryAgent(DictionaryAgent):

    @staticmethod
    def add_cmdline_args(argparser):
        group = DictionaryAgent.add_cmdline_args(argparser)
        group.add_argument(
            '--dict_class', default=class2str(InsultsDictionaryAgent)
        )

    def act(self):
        """Add only words passed in the 'text' field of the observation to this dictionary."""
        text = self.observation.get('text')
        if text:
            self.add_to_dict(self.tokenize(text))
        return {'id': 'InsultsDictionary'}

class InsultsAgent(Agent):

    @staticmethod
    def add_cmdline_args(argparser):
        config.add_cmdline_args(argparser)
        InsultsAgent.dictionary_class().add_cmdline_args(argparser)

    @staticmethod
    def dictionary_class():
        return InsultsDictionaryAgent

    def __init__(self, opt, shared=None):
        self.id = 'InsultsAgent'
        self.episode_done = True
        super().__init__(opt, shared)
        if shared is not None:
            self.is_shared = True
            return
        # Set up params/logging/dicts
        self.is_shared = False

        self.model_name = opt['model_name']

        if self.model_name == 'cnn_word':
            print('create word dict')
            self.word_dict = InsultsAgent.dictionary_class()(opt)
            ## NO EMBEDDINGS NOW
            #print('create embedding matrix')
            #self.embedding_matrix = load_embeddings(opt, self.word_dict.tok2ind)
            self.embedding_matrix = None
            self.num_ngrams = None
        if self.model_name == 'log_reg' or self.model_name == 'svc':
            self.word_dict = None
            self.embedding_matrix = None
            self.num_ngrams = 6

        print('create model', self.model_name)
        self.model = InsultsModel(self.model_name, self.word_dict, self.embedding_matrix, opt)
        self.n_examples = 0
        self.dpath = os.path.join(opt['datapath'], 'insults')
        if (self.model.model_type == 'ngrams' and
                (os.path.isfile(os.path.join(self.dpath, 'ngrams_vect_special.bin')) and
                     os.path.isfile(os.path.join(self.dpath, 'ngrams_vect_general_0.bin')) and
                     os.path.isfile(os.path.join(self.dpath, 'ngrams_vect_general_1.bin')) and
                     os.path.isfile(os.path.join(self.dpath, 'ngrams_vect_general_2.bin')) and
                     os.path.isfile(os.path.join(self.dpath, 'ngrams_vect_general_3.bin')) and
                     os.path.isfile(os.path.join(self.dpath, 'ngrams_vect_general_4.bin')) and
                     os.path.isfile(os.path.join(self.dpath, 'ngrams_vect_general_5.bin'))) ):
            print ('reading vectorizers selectors')
            self.model.vectorizers, self.model.selectors = get_vectorizer_selector(self.dpath, self.num_ngrams)



    def observe(self, observation):
        observation = copy.deepcopy(observation)
        if not self.episode_done:
            # if the last example wasn't the end of an episode, then we need to
            # recall what was said in that example
            prev_dialogue = self.observation['text']
            observation['text'] = prev_dialogue + '\n' + observation['text']
        self.observation = observation
        self.episode_done = observation['episode_done']
        return observation

    def act(self):
        # call batch_act with this batch of one
        return self.batch_act([self.observation])[0]

    def batch_act(self, observations):
        if self.is_shared:
            raise RuntimeError("Parallel act is not supported.")

        batch_size = len(observations)
        # initialize a table of replies with this agent's id
        batch_reply = [{'id': self.getID()} for _ in range(batch_size)]
        examples = [self._build_ex(obs) for obs in observations]
        valid_inds = [i for i in range(batch_size) if examples[i] is not None]
        examples = [ex for ex in examples if ex is not None]

        if 'labels' in observations[0]:
            self.n_examples += len(examples)
            batch = self._batchify_nn(examples)
            predictions = self.model.update(batch)
            predictions = self._predictions2text(predictions)
            for i in range(len(predictions)):
                batch_reply[valid_inds[i]]['text'] = predictions[i]
        else:
            batch = self._batchify_nn(examples)
            predictions = self.model.predict(batch)
            #print ('Predict:', predictions)
            predictions = self._predictions2text(predictions)
            for i in range(len(predictions)):
                batch_reply[valid_inds[i]]['text'] = predictions[i]

        return batch_reply

    def _build_ex(self, ex):
        if 'text' not in ex:
            return
        """Find the token span of the answer in the context for this example.
        """
        inputs = dict()
        inputs['question'] = ex['text']
        if 'labels' in ex:
            inputs['labels'] = ex['labels']

        return inputs

    def _batchify_nn(self, batch):
        question = []
        for ex in batch:
            question.append(self.word_dict.txt2vec(ex['question']))
        question = pad_sequences(question, maxlen=self.model.opt['max_sequence_length'], padding='post')
        if len(batch[0]) == 2:
            y = [1 if ex['labels'][0] == 'Insult' else 0 for ex in batch]
            return question, y
        else:
            return question

    def _predictions2text(self, predictions):
        y = ['Insult' if ex > 0.5 else 'Non-insult' for ex in predictions]
        return y

    def _text2predictions(self, predictions):
        y = [1. if ex == 'Insult' else 0 for ex in predictions]
        return y

    def report(self):
        report = dict()
        report['updates'] = self.model.updates
        report['n_examples'] = self.n_examples
        report['loss'] = self.model.train_loss
        report['accuracy'] = self.model.train_acc
        report['auc'] = self.model.train_auc
        return report

    def save(self):
        self.model.save()


class OneEpochAgent(InsultsAgent):

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.observation = ''
        self.observations = []

    def observe(self, observation):
        observation = copy.deepcopy(observation)
        try:
            prev_dialogue = self.observation['text']
            prev_labels = self.observation['labels']
            observation['text'] = prev_dialogue + '\n' + observation['text']
            observation['labels'] = prev_labels + observation['labels']
            self.observation = observation
        except TypeError:
            self.observation = observation
        except KeyError:
            self.observation = observation
        return observation

    def _batchify_ngrams(self, batch):
        questions = []
        for ex in batch:
            questions.append(ex['question'])

        ngrams_questions = vectorize_select_from_data(questions, self.model.vectorizers, self.model.selectors)
        if len(batch[0]) == 2:
            y = [1 if ex['labels'][0] == 'Insult' else 0 for ex in batch]
            return ngrams_questions, y
        else:
            return ngrams_questions

    def batch_act(self, observations):
        self.observations += observations

        if self.is_shared:
            raise RuntimeError("Parallel act is not supported.")

        batch_size = len(observations)
        # initialize a table of replies with this agent's id
        batch_reply = [{'id': self.getID()} for _ in range(batch_size)]
        examples = [self._build_ex(obs) for obs in observations]
        valid_inds = [i for i in range(batch_size) if examples[i] is not None]
        examples = [ex for ex in examples if ex is not None]

        if 'labels' in observations[0]:
            self.n_examples += len(examples)
        else:
            batch = self._batchify_ngrams(examples)
            predictions = self.model.predict(batch).reshape(-1)
            predictions = self._predictions2text(predictions)
            for i in range(len(predictions)):
                batch_reply[valid_inds[i]]['text'] = predictions[i]

        return batch_reply

    def save(self):
        if not self.is_shared:
            train_data = [observation['text'] for observation in self.observations if 'text' in observation.keys()]
            train_labels = self._text2predictions([observation['labels'][0] for observation in self.observations if 'labels' in observation.keys()])

            # this should be done once for each launch!!!
            if not (os.path.isfile(os.path.join(self.dpath, 'ngrams_vect_special.bin')) and
                    os.path.isfile(os.path.join(self.dpath, 'ngrams_vect_general_0.bin')) and
                    os.path.isfile(os.path.join(self.dpath, 'ngrams_vect_general_1.bin')) and
                    os.path.isfile(os.path.join(self.dpath, 'ngrams_vect_general_2.bin')) and
                    os.path.isfile(os.path.join(self.dpath, 'ngrams_vect_general_3.bin')) and
                    os.path.isfile(os.path.join(self.dpath, 'ngrams_vect_general_4.bin')) and
                    os.path.isfile(os.path.join(self.dpath, 'ngrams_vect_general_5.bin')) ):
                print ('No vectorized data found. Vectorizing train data')
                create_vectorizer_selector(train_data, train_labels, self.dpath,
                                           ngram_list=[1, 2, 3, 4, 5, 3],
                                           max_num_features_list=[2000, 4000, 100, 1000, 1000, 2000],
                                           analyzer_type_list=['word', 'word', 'word', 'char', 'char', 'char'])
            if self.model.vectorizers is None:
                print ('Get vectorizers and selectors')
                self.model.vectorizers, self.model.selectors = get_vectorizer_selector(self.dpath, self.num_ngrams)
            self.model.num_ngrams = self.num_ngrams

            if os.path.isfile(os.path.join(self.dpath, 'train_vectorized.mtx')):
                print('Reading vectorized train dataset')
                X_train = mmread(os.path.join(self.dpath, 'train_vectorized.mtx'))
            else:
                print('Vectorizing train dataset')
                X_train = vectorize_select_from_data(train_data, self.model.vectorizers, self.model.selectors)
                mmwrite(os.path.join(self.dpath, 'train_vectorized'), X_train)

            print('Training model', self.model_name)
            self.model.update([X_train, train_labels])
            #print('Comment', train_data[3951:3952])
            #print('Vectorized:', self.model.predict(X_train.tocsr()[3951:3952,:]))
            #if 1:
            #    comment = 'aimles abe you are a delusion weirdo that should be locked up inside and kept away from computer .. anyone who doubt the lunacy of abe the moronic prog frog from nyc please read thi thread and you will see abe completely come apart from the seam and go off the deep end ..\nexactly .. like , i don not really give a shit how fat a person may or may not feel , but it really i shit when your clothe don not fit .. and not in a "i am so gross" way but in a "i cannot very well leave the house naked ?! ?! " way ..\nthi i someone whose self-importance got the best of him .. if anyone should be fired it would be tate .. thi kind of behavior i inexcusable and unbecom of a man who hold the future of many employee in hi hand ..\nha .. i am glad you are pena i girl , you know he really want to play for the red sox .. he i from boston and went to northeastern but got cut by the sox and he i terrible .. haha .. you have zero valid point ..\ni think it clearly display your lack of rational thought if you truly believe that ..\n@atvcar nobody like the messenger ..\nare you stebber ?! ?!\nwtf kim kwang soo that mostly t-ara money you fuck leech\nso drink at DD i considered "normal" now .. i bet you think you are just such a cool kid eh ?! ?! your generation i so fucked\n@memphisdawg thi i part of what the ncaa wa talk about mak a requirement .. right now , scholarship are one year renewable scholarship , mak it easier for scumbag like saban to release a kid .. some school are go ahead nd offer these four year scholly .. i think the big DD wa push their school to do thi ..\n"what happened to all the talk about that industry hurt because obama wouldn not let them drill all over the place .. " that , i what you call a straw man arguement .. no one i say the "industry i hurting" , because it i not .. what people are say i that it i moronic to send almost a billion dollar a day to country that hate u when we could be develop our own natural resource ..\nlol , your full of chit , just another stupid liberal\nif rcae i your premise - help u .. if fear of religion i your premise - god help u all ?! ?!\ncan you post a statement without us the word idiot ?! ?! try be a little more eloquent with your retort or just start pull the w off of fly to while the hour away ..\ni keep it fresher than the next bitch no need .. for you to ever sweat the next bitch .. with speed , i make the best bitch see the exit .. indeed , you gotta know your thoroughly respected by me , you get the key to the lexu , with no driver you gotcha own \'96 suh-in .. the ride and keep your as tighter than versace that why you gotta watch your friend you got to watch me they conniv shit the first chance to crack the bank they try me , all they get i DD cent frank and papaya , from the village to the tele time to kill it on your belly no question i got more black chick between my sheet than essence they say sex i a weapon , so when i shoot mmet your death in les than DD second still poundin in my after life .. laugin my shit i tight you who askin right .. [choru : ] aint no nigger like the one i got no one can fuck you betta sleep around but he give me alot keep you in diamond and\nthey look like moron \n\\ \\ supposedly , a very rare pattern in the "c" position , or sometime referred to a the "bullish doggy-style manuever" , take your pick ?! ?! \\ \\ lol ?! ?! http : //w .. screencast .. com/users/hoof_hearted/folders/default/media/63e5082f-2692-4214-865c-4701ba356a88very suspect , though ?! ?! (*if someone ha posted thi , my apology , i\\\\ have not seen it)\nit get more and more like butlin a the week go on ..\ni gues you are those who have a fetish for huge silicon boob ..\nyou are a hater .. plian and simple ..\nit fuck massive xd\nnot funny dude .. but i get it ..\nbecause you are so cool , that i why ?! ?!\nthe only th murray i go to beat i hi racket against the ground .. nole in DD easy set ..\nwhat a fool you are , we all know that you think of noth more than be gay ..\nbush ha been long gone .. don not you lib itur d have anyth be tter to cam pair ba ry to ?! ?! if not then shut your ob ummer hol e ..\numm there somth called the internet go to google and type in porn there ya go and gues what the best part i it free ..\ni love you .. not even kidd .. you just made my life whole again .. you are a rock star ..\nhow are the dodger a one man show ?! ?! kemp didn not play DD game and they still kept on winn \ninhofe , there i noth the epa can do to miner what the min company haven not already done and done far worse .. they are the one literally kill the miner for the sake of profit .. they are the one you should be investigat and not protect ..\npeople like you are the reason the economy i fail ..\nwa ?! ?! wa ?! ?! listen to those libturd whine ?! ?! they know they are toast so they yap and snip like little chihauhau ?! ?! (no offense to chihauhau intended) ..\nromney i track record in election i not particularly good when he come face-to-face with effective politician and campaigner (a opposed to has-been and laughingstocks) .. given how rubin demeaned gingrich and santorum dur the primary , she cannot really see romney i nomination a some great achievement ..\nin my opinion , the odd are that , prompted by tel aviv , the neocon , and christian fundamentalist in the pentagon , a romney administration will attack iran and commence a hundred year global religiou war which everyone will lose .. regional alliance , constant combat , and widespread poverty will not be unlike the condition described by george orwell in hi book , "1984 .. "\nwe need a government , for the people , by the people ..\ncadge me a fin .. i will pay you back ..\ncompared with the usa , zimbabwe look clean ..\nmaybe you need a better trained guy\nyou`re a sad excuse for a human be ..\nvery true , mel .. very true .. loyal to the end .. lol\ndick face you would do shit fake as \nkeep tell yourself that , you dumb tea maggot .. the only true racist left are turd like you and your worthles tea maggot friend ..\ni think the sausage might make all the difference .. the white peppered gravy sound fine , it i just the idea of pour it on what appear to be an unsweetened scone and call it a meal seem strange .. so you don not have kipper ?! ?! that seem strange .. i thought that every homestead had a smokehouse and there are herr in the sea .. anyway , i like your scot irish theme .. there might even be an ulster scot right here on my-t ..\nexcellent comment , my thought entirely ..\nyou don not appear to be familiar with the obama campaign , which i a bit surpris ..\nnoynoy aquino i an ugly mother fuck you c k e r ?! ?!\nthe first draft dodger who became president wa bill clinton\nanimal planet told that mermaid are real ..\nobama i a friggin idiot , nuff said\nlemme hole it\nryuzaki ; i like kevin nash .. you don not .. get over it ..\ni will fuck kill you in real life track you down than shove a katana you your as just bcuz we asian r smarted than you dick doesnt mean you could beat him up i swear i will be at chicago in DD hour \nalien DD wa full of metaphor , subtext , and deeper mean too .. still "sucked" .. deep doe not equal smart doe not equal good .. thi i quite fascinat , it like the opposite of that me3 debacle where people would constantly post the same you tube deconstruction of how the game went wrong , despite all the fun they had actually play the game .. in thi case , thi particular writer and the comment do a similar deconstruction , yet he explicitly say "it i still fun tho" and people are still butthurt ..\nall i have to say to you bay i : fuck you ?! ?!\nany arrest in the terror attack at the travel agency ?! ?! ruby , where were you on the night in question ?! ?!\ngay faggot nerd fucker \\r trolololololololololololololloloolololololololololololololol\nmore jurk off material for apple fan boy .. those not liv in mom i basement could care les \nso in your perverted mind if someone post someth about obama that i "discredited" it i ok for you to post the unmitigated bullshit that you do ?! ?! wha ha ha ha\nand why do we need that when practically even the worst secret get though ?! ?!\ni thi another DD % th , senator kyle ?! ?!\n@karla\\ \\ \\ \\ omg ?! ?! i wish i could say i hadn not gone through it , but i have .. i once dated a guy who had lost hi job .. hi fiance had broken up with him because she didn not want to marry an unemployed man .. silly me thought i would be rewarded for be a \'good woman .. \' that i what he alway said to me .. \\ \\ \\ \\ a soon a he got another job he made himself very scarce .. i wa heartbroken .. thank god it now seem like a lifetime ago .. unfortunately th didn not turn out that well for him in the long run though ..\nreally suck isn not the word , when many of our nuclear power plant start melt down , it will literally be hell on earth in the u and we have no one else to blame but our own corrupt government ?! ?!'
            #    print('Comment:', comment)
            #    comment_features = vectorize_select_from_data([comment], self.model.vectorizers, self.model.selectors)
            #    print('Features:', comment_features.tocsr())
            #    print('Predicted:', (self.model.predict(comment_features.tocsr()[0:1,:])))

        print ('\n[model] trained loss = %.4f | acc = %.4f | auc = %.4f' %
               (self.model.train_loss, self.model.train_acc, self.model.train_auc,))
        self.model.save()







