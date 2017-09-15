import copy
import math
import os
import json
import pickle
from parlai.core.agents import create_agent
from parlai.core.params import ParlaiParser
from parlai.core.utils import Timer
from parlai.core.worlds import create_task

from deeppavlov.agents.squad.squad import SquadAgent as Agent

def setup_data(path):
    print('loading: ' + path)
    with open(path) as data_file:
        squad = json.load(data_file)['data']
    for article in squad:
        # each paragraph is a context for the attached questions
        for paragraph in article['paragraphs']:
            # each question is an example
            for qa in paragraph['qas']:
                question = qa['question']
                id = qa['id']
                answers = (a['text'] for a in qa['answers'])
                context = paragraph['context']
                yield {'text': context + '\n' + question, 'episode_done': True, 'id': id}



def main(args=None):
    # Get command line arguments
    parser = ParlaiParser(True, True)
    train = parser.add_argument_group('Scoring Arguments')

    train.add_argument('--datafile', type=str, default='../squad/dev-v1.1.json',
                       help='path to squad dev set')
    train.add_argument('--prediction_file', type=str, default='../squad/predictions',
                       help='where to store file with predictions')

    Agent.add_cmdline_args(parser)
    opt = parser.parse_args(args=args)
    generator = setup_data(opt['datafile'])

    with open(opt['pretrained_model']+'.pkl', 'rb') as f:
        file = pickle.load(f)

    output_file = open(opt['pretrained_model']+'.json', 'w')

    opt = file['config']
    agent = Agent(opt)

    iter = 0
    index = 0
    batch_size = opt['batchsize']
    output = {}

    def process_batch(batch, batch_size, dict):
        batch_reply = agent.batch_act(batch)
        for i in range(batch_size):
            dict[batch[i]['id']] = batch_reply[i]

    for i in generator:
        index+=1

        if iter == 0:
            batch = []

        batch.append(i)
        iter+=1

        if iter==batch_size:
            print('{} examples processed'.format(index))
            process_batch(batch, batch_size, output)
            iter=0

    if iter>0:
        process_batch(batch, len(batch), output)

    json.dump(output, output_file)

if __name__ == '__main__':
    main()
