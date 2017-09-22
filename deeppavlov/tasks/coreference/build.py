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

import parlai.core.build_data as build_data
from parlai.core.utils import Timer
import os
import time
from sklearn import cross_validation
import sys
import json
#reload(sys)
#sys.setdefaultencoding('utf-8')

def RuCoref2CoNLL(path, out_path, language='russian'):
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
    scope_trees = defaultdict(lambda: intervaltree.IntervalTree())
    
    tokens_path = os.path.join(path, ".".join([tokens_fname, tokens_ext]))
    groups_path = os.path.join(path,".".join([groups_fname, groups_ext]))
    doc_count = 1
    print('Convert rucoref corpus into conll format ...')
    start = time.time()
    coref_dict = {}
    with open(groups_path, "r") as groups_file:
        for line in groups_file:
            doc_id, variant, group_id, chain_id, link, shift, lens, content, tk_shifts, attributes, head, hd_shifts = line[:-1].split('\t')       
            if doc_id not in coref_dict:
                coref_dict[doc_id] = {'unos': {},'starts': {},'ends': {}}
                if len(tk_shifts.split(',')) == 1:
                    if tk_shifts not in coref_dict[doc_id]['unos']:
                        coref_dict[doc_id]['unos'][shift] = [chain_id]
                    else:
                        coref_dict[doc_id]['unos'][shift].append(chain_id)
                else:
                    tk = tk_shifts.split(',')
                    if tk[0] not in coref_dict[doc_id]['starts']:
                        coref_dict[doc_id]['starts'][tk[0]] = [chain_id]
                    else:
                        coref_dict[doc_id]['starts'][tk[0]].append(chain_id)
                    if tk[-1] not in coref_dict[doc_id]['ends']:
                        coref_dict[doc_id]['ends'][tk[-1]] = [chain_id]
                    else:
                        coref_dict[doc_id]['ends'][tk[-1]].append(chain_id)
            else:   
                if len(tk_shifts.split(',')) == 1:
                    if tk_shifts not in coref_dict[doc_id]['unos']:
                        coref_dict[doc_id]['unos'][shift] = [chain_id]
                    else:
                        coref_dict[doc_id]['unos'][shift].append(chain_id)
                else:
                    tk = tk_shifts.split(',')
                    if tk[0] not in coref_dict[doc_id]['starts']:
                        coref_dict[doc_id]['starts'][tk[0]] = [chain_id]
                    else:
                        coref_dict[doc_id]['starts'][tk[0]].append(chain_id)
                    if tk[-1] not in coref_dict[doc_id]['ends']:
                        coref_dict[doc_id]['ends'][tk[-1]] = [chain_id]
                    else:
                        coref_dict[doc_id]['ends'][tk[-1]].append(chain_id)
        groups_file.close()
    
    # Write conll structure
    with open(tokens_path, "r") as tokens_file:
        header = next(tokens_file)[:-1]
        k = 0
        for line in tokens_file:
            doc_id, shift, length, token, lemma, gram = line[:-1].split('\t')
            data['word_number'].append(k)
            data['word'].append(token)
            if token == '.':
                k = 0
            else:
                k += 1
            data['doc_id'].append('bc'+doc_id)
            data['part_id'].append(part_id)
            data['lemma'].append(lemma)
            data['part_of_speech'].append(gram[0:-1])
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
            
        tokens_file.close()  
    # Write conll structure in file
    conll = os.path.join(out_path, ".".join([language,'v4_conll']))
    with codecs.open(conll, 'w', encoding='utf-8') as CoNLL:
        for i in tqdm(xrange(len(data['doc_id']))):
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
                                                    data["coref"][i]))
                CoNLL.write('\n')
                CoNLL.write('#end document\n')
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
                                                        data["coref"][i]))
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
                                                        data["coref"][i]))
                    CoNLL.write('\n')
                    CoNLL.write('#end document\n')
                    CoNLL.write('#begin document ({}); part {}\n'.format(data['doc_id'][i+1], data["part_id"][i+1]))
                    
    print('End of convertion. Time - {}'.format(time.time()-start))
    return None

def split_doc(inpath, outpath, language):
    # split massive conll file to many little
    print('Start of splitting ...') 
    with open(inpath, 'r+') as f:
        lines = f.readlines()
        f.close()
    set_ends = []
    k = 0
    print('Splitting conll document ...')
    for i in range(len(lines)):
        if line[i] == '#end document\n':
            set_ends.append([k,i])
            k = i+1
    for i in range(len(set_ends)):
        cpath = os.path.join(outpath, ".".join([str(i), language,'v4_conll']))
        with open(cpath,'w') as c:
            for j in range(set_ends[i][0],set_ends[i][1]):
                c.write(lines[j])
            c.close()
    
    del lines
    print('Splitts {} docs in {}.'.format(len(set_ends),outpath))
    del set_ends
    del k
    
    return None

def train_test_split(inpath,output,split,random_seed):
    z = os.listdir(inpath)
    doc_split = cross_validation.ShuffleSplit(len(z),
                                              n_iter=1,
                                              test_size=split,
                                              random_state=random_seed)
    train_set = [z[i] for i in sorted(list(doc_split)[0][0])]
    test_set = [z[i] for i in sorted(list(doc_split)[0][1])]
    
    train_path = os.path.join(output, 'train')
    test_path = os.path.join(output, 'test')
    
    for x in train_set:
        build_data.move(x, os.path.join(train_path,x))
    for x in test_set:
        build_data.move(x, os.path.join(test_path,x))
    
    return None

def get_all_texts_from_tokens_file(tokens_path, out_path):
    text_count = 0
    lengths = {}
    texts = {}
    # determine number of texts and their lengths
    with open(tokens_path, "r") as tokens_file:
        header = tokens_file.readline()[:-1]
        for line in tokens_file:
            doc_id, shift, length, token, lemma, gram = line[:-1].split('\t')
            doc_id, shift, length = map(int, (doc_id, shift, length))
            lengths[doc_id] = shift + length
    text_count = len(lengths)
    
    texts = {doc_id: [' ']*length for (doc_id, length) in lengths.items()}
    # read texts
    with open(tokens_path, "r") as tokens_file:
        header = tokens_file.readline()[:-1]
        for line in tokens_file:
            doc_id, shift, length, token, lemma, gram = line[:-1].split('\t')
            doc_id, shift, length = map(int, (doc_id, shift, length))
            texts[doc_id][shift:shift + length] = token
    for doc_id in texts:
        texts[doc_id] = "".join(texts[doc_id])
    
    with open(out_path, "w") as out_file:
        for doc_id in texts:
            out_file.write(texts[doc_id])
            out_file.write("\n")
    return None

def get_char_vocab(input_filenames, output_filename):
  vocab = set()
  for filename in input_filenames:
    with open(filename) as f:
      for line in f.readlines():
        for sentence in json.loads(line)["sentences"]:
          for word in sentence:
            vocab.update(word)
  vocab = sorted(list(vocab))
  with open(output_filename, "w") as f:
    for char in vocab:
      f.write(u"{}\n".format(char).encode("utf8"))
  print("[Wrote {} characters to {}] ...".format(len(vocab), output_filename))


def build(opt):
    # get path to data directory
    dpath = os.path.join(opt['datapath'], 'coreference')
    # define version if any
    version = '1.0'
    language = opt['language']
    dpath = os.path.join(dpath, language)
    # check if data had been previously built
    if not build_data.built(dpath, version_string=version):
        print('[building data: ' + dpath + '] ...')

        # make a clean directory if needed
        if build_data.built(dpath):
            # an older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # download the russian datasets, pretrain embeddings.
        url = 'http://rucoref.maimbava.net/files/'  # datasets URL
        embed_url = 'https://drive.google.com/file/d/0B2Ow60cJSy7EWUpSdVoxV2JsTTA' # embeddings url
        scorer_url = 'http://conll.cemantix.org/'
        print('[Download the word embeddings]...')
        
        # download the conll-2012 scorer v 8.1
        build_data.download(scorer_url, os.path.join(dpath,'scorer'), 'download reference-coreference-scorers.v8.01.tar.gz')
        build_data.untar(os.path.join(dpath,'scorer'), 'download reference-coreference-scorers.v8.01.tar.gz')
        os.remove(os.path.join(dpath,'scorer','download reference-coreference-scorers.v8.01.tar.gz'))
        
        # download embeddings
        build_data.download_from_google_drive(embed_url, os.path.join(dpath,'pretrain_embeddings'))
        fname = 'rucoref_29.10.2015.zip'
        start = time.time() # Need rewrite in utils.Timer format
        print('[Download the rucoref dataset]...')
        build_data.download(url, dpath, fname)
        # uncompress it
        build_data.untar(dpath, fname)
        # remove zip file
        os.remove(os.path.join(dpath, fname))
        print('End of download: time - {}'.format(time.time()-start))
        
        # Get pure text from Tokens.txt for creating char dictionary
        build_data.make_dir(os.path.join(dpath,'pure_text'))
        get_all_texts_from_tokens_file(os.path.join(dpath,'rucoref_29.10.2015','Tokens.txt'), os.path.join(dpath,'pure_text','Pure_text.txt'))
        
        # Get char dictionary from pure text
        build_data.make_dir(os.path.join(dpath,'vocab'))
        get_char_vocab(os.path.join(dpath,'pure_text','Pure_text.txt'), os.path.join(dpath,'vocab','char_vocab.{}.txt'.format(language)))
        
        # Convertation rucorpus files in conll files
        conllpath = os.path.join(dpath, 'ru_conll')
        build_data.make_dir(conllpath)
        RuCoref2CoNLL_0(os.path.join(dpath,'rucoref_29.10.2015'), conllpath, language)
        # splits conll files
        start = time.time()
        conlls = os.path.join(dpath, 'ru_conlls')
        build_data.make_dir(conlls)
        split_doc(os.path.join(conllpath, language+'.v4_conll'), conlls, language)
        build_data.remove_dir(conllpath)
        # create train and test partitions
        train_test_split(conlls,dpath,opt['split'],opt['random_seed'])
        build_data.remove_dir(conlls)
        build_data.make_dir(os.path.join(dpath,'report'))
        print('End of data splitting. Time - {}'.format(time.time()-start))
        # mark the data as built
        build_data.mark_done(dpath, version_string=version)
        
        return None
