'''
Reads the senseval-3 supervised word sense disambiguation dataset
'''

from nltk.tokenize import word_tokenize, sent_tokenize
import re
import numpy as np


class KeyDataset(object):
    def __init__(self):
        self.context_m = None
        self.contexts_str = []
        self.contexts_v = []
        self.instance_ids = []
        self.sense_ids = []
    

class DatasetReader(object):
    exp = re.compile('(.*)(<head>[^<]+</head>)(.*)')

    def __init__(self, context_model):
        self.context_model = context_model
       
    def read_dataset(self, contexts_filename, key_filename, train, isolate_target_sentence):
        contexts = self.represent_contexts(contexts_filename, isolate_target_sentence)
        dataset, key2ind, ind2key = self.read_instances(key_filename, contexts, train)
        return dataset, key2ind, ind2key
        
    def next_context(self, contexts_filename):        
        with open(contexts_filename, 'r') as f:
            while True:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    if line.startswith('<context>'):
                        line = f.readline()
                        yield line
                if not line:
                    break
                
    def lower(self, words):
        return [word.lower() for word in words]

    def extract_context(self, text):
        segments = self.exp.match(text)
        if segments is not None:
            seg_left = segments.group(1)
            target_word = segments.group(2)
            seg_right = segments.group(3)
            words_left = self.lower(word_tokenize(seg_left))
            words_right = self.lower(word_tokenize(seg_right))
            words = words_left + [target_word] + words_right
            return words, len(words_left)
        else:
            return None, None


    def extract_target_context(self, paragraph, isolate_target_sentence):
        
        if isolate_target_sentence:
            for sent in sent_tokenize(paragraph):
                words, position = self.extract_context(sent)
                if words is not None:
                    break
        else:
            words, position = self.extract_context(paragraph)
        return words, position

        
    def represent_contexts(self, contexts_filename, isolate_target_sentence):
        '''
        <lexelt item="activate.v">


        <instance id="activate.v.bnc.00008457" docsrc="BNC">
        <context>
        ... and continue to have an important role in <head>activating</head> laity for what are judged to be religious goals both personally and socially . But generally speaking ...  
        </context>
        </instance>
        '''
        contexts_num = 0
        contexts = []
        for context in self.next_context(contexts_filename):
            contexts_num += 1 
            sent_context, position = self.extract_target_context(context, isolate_target_sentence)
            sent_context_str = ' '.join(sent_context[:position]) + ' [' + sent_context[position] + '] ' + ' '.join(sent_context[position+1:])  
            contexts.append((sent_context_str, self.context_model.context2vec(sent_context, position)))
        
        return contexts
    
    
    def read_instances(self, key_filename, contexts, train):
        '''
        activate.v activate.v.bnc.00251499 38201 38202
        activate.v activate.v.bnc.00270989 38201
        activate.v activate.v.bnc.00307829 U
        '''   

        dataset = []
        key2ind = {}
        ind2key = []
        
        last_key = None
        with open(key_filename, 'r') as f:
            for i, line in enumerate(f):
                if len(line.strip()) == 0:
                    continue
                toks = line.strip().split()
                key = toks[0]
                instance_id = toks[1]
                sense_ids = toks[2:]
                if key is not None and key != last_key:
                    if last_key is not None:
                        dataset[-1].context_m = np.vstack(dataset[-1].contexts_v) 
                        norm = np.sqrt((dataset[-1].context_m * dataset[-1].context_m).sum(1))
                        norm[norm==0.] = 1.
                        dataset[-1].context_m /= norm.reshape((norm.shape[0], 1))  # normalize
                   
                    key2ind[key] = len(dataset)
                    ind2key.append(key)
                    dataset.append(KeyDataset())
                    last_key = key
                dataset[-1].contexts_str.append(contexts[i][0])    
                dataset[-1].contexts_v.append(contexts[i][1])
                dataset[-1].instance_ids.append(instance_id)
                if train:
                    dataset[-1].sense_ids.append(sense_ids)

        dataset[-1].context_m = np.vstack(dataset[-1].contexts_v)
        norm = np.sqrt((dataset[-1].context_m * dataset[-1].context_m).sum(1))
        norm[norm==0.] = 1.
        dataset[-1].context_m /= norm.reshape((norm.shape[0], 1))  # normalize
 
        return dataset, key2ind, ind2key       
                
            

if __name__ == '__main__':
    
    class DummyContextModel(object):
        
        def context2vec(self, sent_context, position):
            vec = np.array([1.0/len(sent_context) for _ in range(5)], dtype=np.float32)
            return vec
    
    import sys
    
    if len(sys.argv) < 2:
        sys.stderr.write("Usage: %s <train-filename>\n" % sys.argv[0])
        sys.exit(1)
     
    model = DummyContextModel()    
    reader = DatasetReader(model)
    dataset, key2ind = reader.read_dataset(sys.argv[1], sys.argv[1]+'.key', train=True)
    
    print(key2ind)
    for oneset in dataset:
        print()
        print(oneset.context_m)
        print(oneset.contexts_str)
        print(oneset.contexts_v)
        print(oneset.instance_ids)
        print(oneset.sense_ids)
        