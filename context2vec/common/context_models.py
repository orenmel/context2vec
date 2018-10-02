import math

import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L

from context2vec.common.defs import Toks


class CbowContext(object):
    
    """
    Continuous Bag of Words (CBOW) context representation, also called Average of Word Embeddings (AWE).
    Based on word embeddings learned by 3rd-party
    """
    
    def __init__(self, targets, contexts, word2index, stopwords, window_size, word_counts):
        self.targets = targets
        self.contexts = contexts if contexts is not None else targets # if contexts not provided then using same embeddings to represent both target and context words
        self.word2index = word2index
        self.window_size = window_size
        self.stopwords = stopwords       
        self.idf = self.count2idf(word_counts, word2index) if word_counts is not None else None
        
    def context2vec(self, sent_words, position):
        '''
        Convert sentential context into a vector representation
        :param sent_words: a list of words
        :param position: the position of the target slot in sent_words (value of sent_words[i] will be ignored)
        :return vector representation of context
        '''
        
        bow = self.extract_window(sent_words, position)
        bow_inds = [self.word2index[word] for word in bow if word in self.word2index and word not in self.stopwords ]
        if len(bow_inds) == 0:
            print("NOTICE: Empty bow context for: " + str(sent_words))
            print("Trying with stopwords")
            bow_inds = [self.word2index[word] for word in bow if word in self.word2index ]           
        return self.context_rep(bow_inds)
    
    def count2idf(self, word_counts, word2index):
        sum_counts = sum(word_counts.values())
        idf = np.zeros((len(word2index),1), dtype=float)
        for word, count in word_counts.items():
            if word in word2index:
                idf[word2index[word],0] = math.log(float(sum_counts)/count)
        return idf
        
    def extract_window(self, sent_words, position):
        if self.window_size == 0:
            begin = 0
            end = len(sent_words)
        else:
            begin = max(position-self.window_size, 0)
            end = min(position+self.window_size+1, len(sent_words))
        return sent_words[begin:position] + sent_words[position+1:end]         
        
    def context_rep(self, bow_inds):
        if self.idf is None:
            return np.average(self.contexts[bow_inds,:], axis=0)
        else:
            return np.average(self.contexts[bow_inds,:]*self.idf[bow_inds,:], axis=0)
        
    

class BiLstmContext(chainer.Chain):

    """
    Bidirectional LSTM context.
    """
      
    def __init__(self, deep, gpu, word2index, in_units, hidden_units, out_units, loss_func, train, drop_ratio=0.0):
        n_vocab = len(word2index)        
        l2r_embedding=L.EmbedID(n_vocab, in_units)
        r2l_embedding=L.EmbedID(n_vocab, in_units)
        
        if deep:
            super(BiLstmContext, self).__init__(
                l2r_embed=l2r_embedding,
                r2l_embed=r2l_embedding,
                loss_func=loss_func,
                l2r_1 = L.LSTM(in_units, hidden_units),
                r2l_1 = L.LSTM(in_units, hidden_units),
                l3 = L.Linear(2*hidden_units, 2*hidden_units),
                l4 = L.Linear(2*hidden_units, out_units),
            )
        else:
            super(BiLstmContext, self).__init__(
                l2r_embed=l2r_embedding,
                r2l_embed=r2l_embedding,
                loss_func=loss_func,
                l2r_1 = L.LSTM(in_units, hidden_units),
                r2l_1 = L.LSTM(in_units, hidden_units),
                lp_l2r = L.Linear(hidden_units, int(out_units/2)),
                lp_r2l = L.Linear(hidden_units, int(out_units/2))
                
            )
        if gpu >=0:
            self.to_gpu()
        l2r_embedding.W.data = self.xp.random.normal(0, math.sqrt(1. / l2r_embedding.W.data.shape[0]), l2r_embedding.W.data.shape).astype(np.float32)       
        r2l_embedding.W.data = self.xp.random.normal(0, math.sqrt(1. / r2l_embedding.W.data.shape[0]), r2l_embedding.W.data.shape).astype(np.float32)
        
        self.word2index = word2index
        self.train = train
        self.deep = deep
        self.drop_ratio = drop_ratio
        
    def context2vec(self, sent_words, position):
        '''
        Convert sentential context into a vector representation
        :param sent_words: a list of words
        :param position: the position of the target slot in sent_words (value of sent_words[i] will be ignored)
        :return vector representation of context
        '''
        sent = [self.word2index[word] if word in self.word2index else Toks.UNK for word in sent_words]
        return self.context_rep(sent, position)
       
    def __call__(self, sent):
        '''
        Train the network
        :param sent: a minibatch of sentences
        '''
        self.reset_state()
        return self._calculate_loss(sent)
        
    def reset_state(self):
        self.l2r_1.reset_state()
        self.r2l_1.reset_state()     
                     
    def context_rep(self, sent, position):
        self.reset_state()
        sent = sent
        sent = [sent] # used for eval - no minibatching here for now
        sent_arr = self.xp.asarray(sent, dtype=np.int32)
        sent_y = self._contexts_rep(sent_arr)
        return sent_y[position].data[0]           
        
    def _contexts_rep(self, sent_arr):
        
        batchsize = len(sent_arr)
        
        bos = self.xp.full((batchsize,1), Toks.BOS, dtype=np.int32)
        eos = self.xp.full((batchsize,1), Toks.EOS, dtype=np.int32)
                
        l2r_sent = self.xp.hstack((bos,sent_arr))            # <bos> a b c
        r2l_sent = self.xp.hstack((eos,sent_arr[:,::-1]))    # <eos> c b a 

        # generate left-to-right contexts representations
        l2r_sent_h = []
        for i in range(l2r_sent.shape[1]-1): # we don't need the last word in the sentence
            c = chainer.Variable(l2r_sent[:,i])
            e = self.l2r_embed(c)            
            if self.drop_ratio > 0.0:
                h = self.l2r_1(F.dropout(e, ratio=self.drop_ratio, train=self.train))
            else:
                h = self.l2r_1(e)
            l2r_sent_h.append(h)
            
        # generate right-to-left contexts representations
        r2l_sent_h = []
        for i in range(r2l_sent.shape[1]-1): # we don't want the last word in the sentence
            c = chainer.Variable(r2l_sent[:,i])
            e = self.r2l_embed(c)
            if self.drop_ratio > 0.0:
                h = self.r2l_1(F.dropout(e, ratio=self.drop_ratio, train=self.train))
            else:
                h = self.r2l_1(e)
            r2l_sent_h.append(h)
            
        r2l_sent_h.reverse()
        
        # l2r_sent_h: h(<bos>)  h(a)  h(b)
        # r2l_sent_h:   h(b)    h(c) h(<eos>)
        
        # concat left-to-right with right-to-left
        sent_bi_h = []
        for l2r_h, r2l_h in zip(l2r_sent_h, r2l_sent_h):
            if not self.deep: # projecting hidden state to half out-units dimensionality before concatenating
                if self.drop_ratio > 0.0:
                    l2r_h = self.lp_l2r(F.dropout(l2r_h, ratio=self.drop_ratio, train=self.train))
                    r2l_h = self.lp_r2l(F.dropout(r2l_h, ratio=self.drop_ratio, train=self.train))
                else:
                    l2r_h = self.lp_l2r(l2r_h)
                    r2l_h = self.lp_r2l(r2l_h)
            bi_h = F.concat((l2r_h, r2l_h)) # TODO - is concat slow??
            sent_bi_h.append(bi_h)

        
        # Use a 2-layer perceptron to merge the hidden states of both sides of the context
        if self.deep:    
            sent_y = []
            for bi_h in sent_bi_h:
                if self.drop_ratio > 0.0:
                    h1 = F.relu(self.l3(F.dropout(bi_h, ratio=self.drop_ratio, train=self.train)))
                    y = self.l4(F.dropout(h1, ratio=self.drop_ratio, train=self.train))
                else:
                    h1 = F.relu(self.l3(bi_h))
                    y = self.l4(h1)

                sent_y.append(y)
            return sent_y
        else:
            return sent_bi_h

               
    def _calculate_loss(self, sent):
        # sent is a batch of sentences.
        sent_arr = self.xp.asarray(sent, dtype=np.int32)

        sent_y = self._contexts_rep(sent_arr)
        
        sent_x = []
        for i in range(sent_arr.shape[1]):
            x = chainer.Variable(sent_arr[:,i])
            sent_x.append(x)
            
        accum_loss = None
        for y,x in zip(sent_y, sent_x):
            loss = self.loss_func(y, x)
            accum_loss = accum_loss + loss if accum_loss is not None else loss 
        
        return accum_loss

