import os
from tqdm import tqdm
import time
import numpy as np
import json

def score(scorer, keys_path, predicts_path):
    key_files = list(filter(lambda x: x.endswith('conll'), os.listdir(keys_path)))
    pred_files = list(filter(lambda x: x.endswith('conll'), os.listdir(predicts_path)))
    
    for file in tqdm(pred_files):
        predict_file = os.path.join(predicts_path, file)
        gold_file = os.path.join(keys_path, file)
        for metric in ['muc', 'bcub', 'ceafm', 'ceafe']:
            out_pred_score = '{0}.{1}'.format(predict_file, metric)
            cmd = '{0} {1} {2} {3} none > {4}'.format(scorer, metric, gold_file, predict_file, out_pred_score)
            #print(cmd)
            os.system(cmd)

    # make sure that all files processed
    time.sleep(1)

    k = 0
    results = dict()
    res = dict()

    f1=[]
    for metric in ['muc', 'bcub', 'ceafm', 'ceafe']:
        recall = []
        precision = []
        for file in pred_files:
            out_pred_score = '{0}.{1}'.format(os.path.join(predicts_path, file), metric)
            with open(out_pred_score, 'r', encoding='utf8') as score_file:
                lines = score_file.readlines()
                if lines[-1].strip() != '--------------------------------------------------------------------------':
                    continue

                coreference_scores_line = lines[-2]
                tokens = coreference_scores_line.replace('\t', ' ').split()
                r1 = float(tokens[2].strip('()'))
                r2 = float(tokens[4].strip('()'))
                p1 = float(tokens[7].strip('()'))
                p2 = float(tokens[9].strip('()'))
                if r2 == 0 or p2 == 0:
                    continue
                recall.append((r1, r2))
                precision.append((p1, p2))
                k += 1

        r1 = sum(map(lambda x: x[0], recall))
        r2 = sum(map(lambda x: x[1], recall))
        p1 = sum(map(lambda x: x[0], precision))
        p2 = sum(map(lambda x: x[1], precision))
        
        
        r = 0 if r2 == 0 else r1 / float(r2)
        p = 0 if p2 ==0 else p1 / float(p2)
        f = 0 if (p+r) == 0 else (2 * p * r) / (p + r)
        
        
        f1.append(f)
        res[metric] = '{0} precision: ({1:.3f}/{2}) {3:.3f}\t recall: ({4:.3f}/{5}) {6:.3f}\t F-1: {7:.5f}'.format(metric, p1, p2, p, r1, r2, r, f)
        results[metric] = {'p': p, 'r': r, 'f-1': f}

    # muc bcub ceafe
    conllf1 = np.mean(f1[:2] + f1[-1:]) # wtf
    res['using'] = 'using {}/{}'.format(k, 4 * len(key_files)) 
    res['avg-F-1'] = np.mean(f1)
    res['conll-F-1'] = conllf1
    json.dump(results, open(os.path.join(predicts_path, 'results.json'), 'w'))
    return res