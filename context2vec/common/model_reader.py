import numpy
from chainer import cuda
import chainer.serializers as S
import chainer.links as L
from nltk.corpus import stopwords

from .context_models import CbowContext, BiLstmContext
from .defs import IN_TO_OUT_UNITS_RATIO, NEGATIVE_SAMPLING_NUM


class ModelReader(object):
    '''
    Reads a pre-trained model using a config file
    '''

    def __init__(self, config_file):
        self.gpu = -1 # todo support gpu
        print('Reading config file: ' + config_file)
        params = self.read_config_file(config_file)
        print('Config: ', params)
        self.w, self.word2index, self.index2word, self.model = self.read_model(params)
        


    def read_config_file(self, filename):
        
        params = {}
        config_path = filename[:filename.rfind('/')+1]
        params['config_path'] = config_path
        with open(filename, 'r') as f:        
            for line in f:
                if not line.startswith('#'):
                    [param, val] = line.strip().split()
                    params[param] = val
        return params
    
    
    def read_model(self, params, train=False):
        
        if 'model_type' in params:
            model_type = params['model_type']
        else:
            model_type = 'lstm_context'
            
        if model_type == 'lstm_context':
            return self.read_lstm_model(params, train)
        elif model_type == 'bow_context':
            return self.read_bow_model(params)
        else:
            raise Exception("Unknown model type: " + model_type)
        

    def read_lstm_model(self, params, train):
        
        assert train == False # reading a model to continue training is currently not supported
        
        words_file = params['config_path'] + params['words_file']
        model_file = params['config_path'] + params['model_file']
        unit = int(params['unit'])
        deep = (params['deep'] == 'yes')
        drop_ratio = float(params['drop_ratio'])
        
        #read and normalize target word embeddings
        w, word2index, index2word = self.read_words(words_file) 
        s = numpy.sqrt((w * w).sum(1))
        s[s==0.] = 1.
        w /= s.reshape((s.shape[0], 1))  # normalize
        
        context_word_units = unit
        lstm_hidden_units = IN_TO_OUT_UNITS_RATIO*unit
        target_word_units = IN_TO_OUT_UNITS_RATIO*unit
        
        cs = [1 for _ in range(len(word2index))] # dummy word counts - not used for eval
        loss_func = L.NegativeSampling(target_word_units, cs, NEGATIVE_SAMPLING_NUM) # dummy loss func - not used for eval
        
        model = BiLstmContext(deep, self.gpu, word2index, context_word_units, lstm_hidden_units, target_word_units, loss_func, train, drop_ratio)
        S.load_npz(model_file, model)
        
        return w, word2index, index2word, model
    
    def read_bow_model(self, params):
        words_file = params['config_path'] + params['words_file']
        contexts_file = params['config_path'] + params['contexts_file'] if 'contexts_file' in params else None 
        window_size = int(params['window_size'])
        use_stopwords = params['stopwords']
        
        if 'word_counts_file' in params:
            word_counts_file = params['config_path'] + params['word_counts_file']
        else:
            word_counts_file = None
            
        if use_stopwords == 'yes':
            stop = set(stopwords.words('english') + ['.',',','(',')','[',']',':','"',"'","'s","-",';','?','!','|','%','/','\\'])
        else:
            stop = set()
        
        word_counts = self.read_word_counts(word_counts_file) if word_counts_file is not None else None
        
        # read and normalize target words embeddings
        w, word2index, index2word = self.read_words(words_file)
        s = numpy.sqrt((w * w).sum(1))
        s[s==0.] = 1.
        w /= s.reshape((s.shape[0], 1))  # normalize
        
        # read and normalize context words embeddings (if using different embeddings for context words)
        if contexts_file is not None:
            c, _, _ = self.read_words(words_file) # assuming words and contexts vocabs are identical
            s = numpy.sqrt((c * c).sum(1))
            s[s==0.] = 1.
            c /= s.reshape((s.shape[0], 1))  # normalize
        else:
            c = None
            
        model = CbowContext(w, c, word2index, stop, window_size, word_counts)
        
        return w, word2index, index2word, model


    def read_words(self, filename):
        with open(filename, 'r') as f:
            ss = f.readline().split()
            n_vocab, n_units = int(ss[0]), int(ss[1])
            word2index = {}
            index2word = []
            w = numpy.empty((n_vocab, n_units), dtype=numpy.float32)
            for i, line in enumerate(f):
                ss = line.split()
                assert len(ss) == n_units + 1
                word = ss[0]
                word2index[word] = i
                index2word.append(word)
                w[i] = numpy.array([float(s) for s in ss[1:]], dtype=numpy.float32)
        return w, word2index, index2word
    
    def read_word_counts(self, filename):
        counts = {}
        with open(filename) as f:
            for line in f:
                if len(line) > 0:
                    tokens = line.split('\t') 
                    word = tokens[0].strip() 
                    count = int(tokens[1].strip())
                    counts[word] = count
        return counts

    



        