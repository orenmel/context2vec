'''
Evaluates context2vec on the senseval-3 supervised word sense disambiguation benchmark
'''

import sys
import numpy as np


from knn import Knn
from dataset_reader import DatasetReader
from context2vec.common.model_reader import ModelReader

class DummyContextModel(object):
    
    def context2vec(self, sent_context, position):
        vec = np.array([0 for _ in range(100)], dtype=np.float32)
        vec[100 % position] = 1.0
        return vec


if __name__ == '__main__':
    
    if len(sys.argv) < 6:
        print("Usage: %s <train-filename> <test-filename> <result-filename> <model-params> <k> [paragraph]"  % (sys.argv[0]), file=sys.stderr)
        sys.exit(1)
        
    train_filename = sys.argv[1]
    test_filename = sys.argv[2]
    result_filename = sys.argv[3]
    model_params_filename = sys.argv[4]
    k = int(sys.argv[5])
    
    if len(sys.argv) > 6:
        isolate_target_sentence = False
        print("Paragraph contexts")
    else:
        isolate_target_sentence = True
        print("Sentence contexts")    
    
    if train_filename == test_filename:
        print('Dev run.')
        ignore_closest = True
    else:
        ignore_closest = False
        
    debug = False
    dummy = False
    
    print('Reading model..')
    if not dummy:
        model_reader = ModelReader(model_params_filename)
        model = model_reader.model
    else:
        model = DummyContextModel()
    dataset_reader = DatasetReader(model)
    
    print('Reading train dataset..')
    train_set, train_key2ind, train_ind2key = dataset_reader.read_dataset(train_filename, train_filename+'.key', True, isolate_target_sentence)
    knn = Knn(k, train_set, train_key2ind)
    
    print('Reading test dataset..')
    test_set, test_key2ind, test_ind2key = dataset_reader.read_dataset(test_filename, test_filename+'.key', False, isolate_target_sentence)
    
    print('Starting to classify test set:')
    with open(result_filename, 'w') as o:
        for ind, key_set in enumerate(test_set):
            key = test_ind2key[ind]
            if debug:
                print('KEY:', key)
                print()
            for instance_id, vec, text in zip(key_set.instance_ids, key_set.context_m, key_set.contexts_str):
                if debug:
                    print('QUERY:', text.strip())
                result = knn.classify(key, vec, ignore_closest, debug)
                if debug:
                    print()
                #brother.n 00006 501566/0.5 501573/0.4 503751/0.1
                result_line = key + ' ' + instance_id
                for sid, weight in result.items():
                    result_line += ' {}/{:.4f}'.format(sid, weight)
                    
                o.write(result_line+'\n')
                if debug:
                    print('LABELS FOUND: ', result_line)
                    print()
            
        
    
    
    
    
    
    